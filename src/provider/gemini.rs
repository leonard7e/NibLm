use super::{LlmProvider, PromptPart};
use anyhow::{Result, anyhow, ensure};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::thinking::ThinkingLevel;

pub struct GeminiProvider {
    api_key: String,
    client: Client,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::builder()
                .user_agent(super::USER_AGENT)
                .build()
                .unwrap_or_default(),
            base_url: "https://generativelanguage.googleapis.com".to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_base_url(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            client: Client::builder()
                .user_agent(super::USER_AGENT)
                .build()
                .unwrap_or_default(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    #[serde(rename = "systemInstruction", skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Serialize)]
struct GeminiGenerationConfig {
    #[serde(rename = "thinkingConfig", skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GeminiThinkingConfig>,
    // Sampling parameters `temperature`, `top_p`, and `top_k` are deliberately omitted
    // to adhere to Gemini API specs and prevent 400 INVALID_ARGUMENT errors.
}

#[derive(Serialize)]
struct GeminiThinkingConfig {
    #[serde(rename = "thinking_level")]
    thinking_level: &'static str,
}

#[derive(Serialize, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Deserialize)]
struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<GeminiInlineData>,
}

#[derive(Serialize, Deserialize)]
struct GeminiInlineData {
    mime_type: String,
    data: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    error: Option<GeminiError>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
}

#[derive(Deserialize)]
struct GeminiError {
    message: String,
}

/// Maps a numeric integer token budget to an appropriate `thinking_level`.
fn map_budget_to_thinking_level(budget: u32) -> &'static str {
    match budget {
        0 => "minimal",
        1..=2_048 => "low",
        2_049..=16_384 => "medium",
        _ => "high",
    }
}

/// Resolves a `ThinkingLevel` into the corresponding Gemini `thinking_level` string value.
/// Returns `None` when thinking is disabled.
/// If a numeric integer (token budget) is provided, a non-fatal deprecation warning is emitted
/// to stderr and the budget is mapped to an appropriate thinking_level.
fn resolve_thinking_level(thinking: &ThinkingLevel) -> Option<&'static str> {
    match thinking {
        ThinkingLevel::Off => None,
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High => Some("high"),
        ThinkingLevel::Custom(budget) => {
            eprintln!(
                "Warning: Numeric thinking budget is deprecated for Gemini models and will be ignored/mapped"
            );
            Some(map_budget_to_thinking_level(*budget))
        }
    }
}

/// Constructs the `GeminiGenerationConfig` for a given `ThinkingLevel`.
/// Returns `None` if thinking is disabled (`ThinkingLevel::Off`).
fn resolve_generation_config(thinking: &ThinkingLevel) -> Option<GeminiGenerationConfig> {
    resolve_thinking_level(thinking).map(|level| GeminiGenerationConfig {
        thinking_config: Some(GeminiThinkingConfig {
            thinking_level: level,
        }),
    })
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    async fn complete(
        &self,
        system_instruction: &str,
        prompt_parts: &[PromptPart],
        model: &str,
        thinking: &ThinkingLevel,
    ) -> Result<String> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.base_url, model, self.api_key
        );

        let parts: Vec<GeminiPart> = prompt_parts
            .iter()
            .map(|p| match p {
                PromptPart::Text(t) => GeminiPart {
                    text: Some(t.clone()),
                    inline_data: None,
                },
                PromptPart::Image { mime_type, data } => GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: mime_type.clone(),
                        data: STANDARD.encode(data),
                    }),
                },
                PromptPart::Audio { mime_type, data } => GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: mime_type.clone(),
                        data: STANDARD.encode(data),
                    }),
                },
                PromptPart::Video { mime_type, data } => GeminiPart {
                    text: None,
                    inline_data: Some(GeminiInlineData {
                        mime_type: mime_type.clone(),
                        data: STANDARD.encode(data),
                    }),
                },
            })
            .collect();

        let sys_instr = if !system_instruction.is_empty() {
            Some(GeminiContent {
                parts: vec![GeminiPart {
                    text: Some(system_instruction.to_string()),
                    inline_data: None,
                }],
            })
        } else {
            None
        };

        let generation_config = resolve_generation_config(thinking);

        let req_body = GeminiRequest {
            system_instruction: sys_instr,
            contents: vec![GeminiContent { parts }],
            generation_config,
        };

        let res = self.client.post(&url).json(&req_body).send().await?;

        let status = res.status();
        ensure!(
            status.is_success(),
            anyhow!(
                "Gemini API HTTP Error ({}): {}",
                status,
                res.text().await.unwrap_or_default()
            )
        );

        let resp: GeminiResponse = res.json().await?;

        ensure!(
            resp.error.is_none(),
            anyhow!(
                "Gemini API Error ({}): {}",
                status,
                resp.error.unwrap().message
            )
        );

        let text = resp
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content.parts.into_iter().next())
            .and_then(|p| p.text)
            .ok_or_else(|| anyhow!("Invalid or empty Gemini response"))?;

        Ok(text)
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            models: Vec<ModelInfo>,
        }
        #[derive(Deserialize)]
        struct ModelInfo {
            name: String,
        }

        let url = format!("{}/v1beta/models?key={}", self.base_url, self.api_key);

        let res: ModelsResponse = self.client.get(&url).send().await?.json().await?;

        Ok(res
            .models
            .into_iter()
            .map(|m| m.name.replace("models/", ""))
            .collect())
    }

    async fn get_context_limit(&self, model: &str) -> Result<usize> {
        #[derive(Deserialize)]
        struct ModelInfo {
            #[serde(rename = "inputTokenLimit")]
            input_token_limit: Option<usize>,
        }

        let url = format!(
            "{}/v1beta/models/{}?key={}",
            self.base_url, model, self.api_key
        );

        let res = self.client.get(&url).send().await?;
        if !res.status().is_success() {
            return Ok(32768); // Fallback
        }

        let info: ModelInfo = res.json().await?;
        Ok(info
            .input_token_limit
            .unwrap_or(crate::provider::DEFAULT_CONTEXT_LIMIT))
    }

    async fn supports_images(&self, _model: &str) -> Result<bool> {
        Ok(true)
    }
    async fn supports_audio(&self, _model: &str) -> Result<bool> {
        Ok(true)
    }
    async fn supports_video(&self, _model: &str) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Server;

    #[test]
    fn test_gemini_thinking_config_off() {
        let config = resolve_generation_config(&ThinkingLevel::Off);
        assert!(config.is_none());
    }

    #[test]
    fn test_gemini_thinking_config_low() {
        let config = resolve_generation_config(&ThinkingLevel::Low);
        assert_eq!(
            config.unwrap().thinking_config.unwrap().thinking_level,
            "low"
        );
    }

    #[test]
    fn test_gemini_thinking_config_medium() {
        let config = resolve_generation_config(&ThinkingLevel::Medium);
        assert_eq!(
            config.unwrap().thinking_config.unwrap().thinking_level,
            "medium"
        );
    }

    #[test]
    fn test_gemini_thinking_config_high() {
        let config = resolve_generation_config(&ThinkingLevel::High);
        assert_eq!(
            config.unwrap().thinking_config.unwrap().thinking_level,
            "high"
        );
    }

    #[test]
    fn test_gemini_thinking_config_custom() {
        // Budget 0 falls back to "minimal"
        let config_0 = resolve_generation_config(&ThinkingLevel::Custom(0));
        assert_eq!(
            config_0.unwrap().thinking_config.unwrap().thinking_level,
            "minimal"
        );

        // Budget <= 2048 falls back to "low"
        let config_low = resolve_generation_config(&ThinkingLevel::Custom(1024));
        assert_eq!(
            config_low.unwrap().thinking_config.unwrap().thinking_level,
            "low"
        );

        // Budget between 2049 and 16384 falls back to "medium"
        let config_med = resolve_generation_config(&ThinkingLevel::Custom(4096));
        assert_eq!(
            config_med.unwrap().thinking_config.unwrap().thinking_level,
            "medium"
        );

        // Budget > 16384 falls back to "high"
        let config_high = resolve_generation_config(&ThinkingLevel::Custom(32_000));
        assert_eq!(
            config_high.unwrap().thinking_config.unwrap().thinking_level,
            "high"
        );
    }

    #[test]
    fn test_gemini_request_serialization_omits_deprecated_sampling_params() {
        let req = GeminiRequest {
            system_instruction: None,
            contents: vec![GeminiContent { parts: vec![] }],
            generation_config: resolve_generation_config(&ThinkingLevel::Low),
        };

        let json_val = serde_json::to_value(&req).expect("Serialization failed");

        // Verify thinking_level is present inside thinkingConfig
        let gen_config = json_val
            .get("generationConfig")
            .expect("generationConfig should be present");
        let thinking_config = gen_config
            .get("thinkingConfig")
            .expect("thinkingConfig should be present");
        assert_eq!(
            thinking_config
                .get("thinking_level")
                .and_then(|v| v.as_str()),
            Some("low")
        );

        // Verify thinkingBudget is NOT present
        assert!(thinking_config.get("thinkingBudget").is_none());

        // Verify deprecated sampling parameters (temperature, top_p, top_k) are omitted entirely
        assert!(json_val.get("temperature").is_none());
        assert!(json_val.get("top_p").is_none());
        assert!(json_val.get("top_k").is_none());
        assert!(gen_config.get("temperature").is_none());
        assert!(gen_config.get("top_p").is_none());
        assert!(gen_config.get("top_k").is_none());
    }

    #[tokio::test]
    async fn test_gemini_complete() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = GeminiProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock(
                "POST",
                "/v1beta/models/test-model:generateContent?key=test_key",
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": "Zusammenfassung" }
                            ]
                        }
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model", &ThinkingLevel::Off)
            .await
            .unwrap();
        assert_eq!(result, "Zusammenfassung");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_gemini_complete_with_thinking() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = GeminiProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock(
                "POST",
                "/v1beta/models/test-model:generateContent?key=test_key",
            )
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"generationConfig":{"thinkingConfig":{"thinking_level":"low"}}}"#.to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": "Response with thinking" }
                            ]
                        }
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model", &ThinkingLevel::Low)
            .await
            .unwrap();
        assert_eq!(result, "Response with thinking");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_gemini_complete_with_custom_budget_fallback() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = GeminiProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock(
                "POST",
                "/v1beta/models/test-model:generateContent?key=test_key",
            )
            .match_body(mockito::Matcher::PartialJsonString(
                r#"{"generationConfig":{"thinkingConfig":{"thinking_level":"medium"}}}"#.to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                { "text": "Response with fallback" }
                            ]
                        }
                    }
                ]
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model", &ThinkingLevel::Custom(4096))
            .await
            .unwrap();
        assert_eq!(result, "Response with fallback");
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn test_gemini_api_error() {
        let mut server = Server::new_async().await;
        let url = server.url();
        let provider = GeminiProvider::with_base_url("test_key".to_string(), url);

        let mock = server
            .mock(
                "POST",
                "/v1beta/models/test-model:generateContent?key=test_key",
            )
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                "error": {
                    "message": "Invalid request"
                }
            }"#,
            )
            .create_async()
            .await;

        let result = provider
            .complete("", &[PromptPart::Text("Prompt".to_string())], "test-model", &ThinkingLevel::Off)
            .await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Gemini API HTTP Error"));
        assert!(err_msg.contains("Invalid request"));
        mock.assert_async().await;
    }
}
