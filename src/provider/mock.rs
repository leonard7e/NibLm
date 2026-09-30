use crate::provider::{LlmProvider, PromptPart};
use crate::thinking::ThinkingLevel;
use anyhow::Result;
use async_trait::async_trait;

/// A mock implementation of `LlmProvider` for testing purposes.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MockProvider {
    pub supports_images: bool,
    pub supports_audio: bool,
    pub supports_video: bool,
    pub context_limit: usize,
    pub completion_response: String,
    pub models: Vec<String>,
}

impl Default for MockProvider {
    fn default() -> Self {
        Self {
            supports_images: true,
            supports_audio: false,
            supports_video: false,
            context_limit: 8192,
            completion_response: String::new(),
            models: Vec::new(),
        }
    }
}

#[allow(dead_code)]
impl MockProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_supports_images(mut self, supports: bool) -> Self {
        self.supports_images = supports;
        self
    }

    pub fn with_supports_audio(mut self, supports: bool) -> Self {
        self.supports_audio = supports;
        self
    }

    pub fn with_supports_video(mut self, supports: bool) -> Self {
        self.supports_video = supports;
        self
    }

    pub fn with_context_limit(mut self, limit: usize) -> Self {
        self.context_limit = limit;
        self
    }

    pub fn with_completion_response(mut self, response: impl Into<String>) -> Self {
        self.completion_response = response.into();
        self
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn complete(
        &self,
        _system_instruction: &str,
        _user_parts: &[PromptPart],
        _model: &str,
        _thinking: &ThinkingLevel,
    ) -> Result<String> {
        Ok(self.completion_response.clone())
    }

    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(self.models.clone())
    }

    async fn get_context_limit(&self, _model: &str) -> Result<usize> {
        Ok(self.context_limit)
    }

    async fn supports_images(&self, _model: &str) -> Result<bool> {
        Ok(self.supports_images)
    }

    async fn supports_audio(&self, _model: &str) -> Result<bool> {
        Ok(self.supports_audio)
    }

    async fn supports_video(&self, _model: &str) -> Result<bool> {
        Ok(self.supports_video)
    }
}
