use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Selects the reasoning/thinking effort for providers that support it.
/// `Custom(u32)` holds a raw token budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Low,
    Medium,
    High,
    /// A raw token budget (provider-specific meaning).
    Custom(u32),
}

impl ThinkingLevel {
    /// Maps the level to a concrete token budget.
    /// Returns `None` when thinking should be disabled.
    pub fn token_budget(&self) -> Option<u32> {
        match self {
            ThinkingLevel::Off => None,
            ThinkingLevel::Low => Some(1_024),
            ThinkingLevel::Medium => Some(8_192),
            ThinkingLevel::High => Some(32_000),
            ThinkingLevel::Custom(n) => Some(*n),
        }
    }
}

impl Default for ThinkingLevel {
    fn default() -> Self {
        ThinkingLevel::Off
    }
}

impl std::str::FromStr for ThinkingLevel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "off" => Ok(ThinkingLevel::Off),
            "low" => Ok(ThinkingLevel::Low),
            "medium" => Ok(ThinkingLevel::Medium),
            "high" => Ok(ThinkingLevel::High),
            _ => s.parse::<u32>().map(ThinkingLevel::Custom).map_err(|_| {
                anyhow::anyhow!(
                    "Invalid thinking level '{}'. Expected one of: off, low, medium, high, or a non-negative integer token budget.",
                    s
                )
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_token_budget_off() {
        assert_eq!(ThinkingLevel::Off.token_budget(), None);
    }

    #[test]
    fn test_token_budget_levels() {
        assert_eq!(ThinkingLevel::Low.token_budget(), Some(1_024));
        assert_eq!(ThinkingLevel::Medium.token_budget(), Some(8_192));
        assert_eq!(ThinkingLevel::High.token_budget(), Some(32_000));
    }

    #[test]
    fn test_token_budget_custom() {
        assert_eq!(ThinkingLevel::Custom(500).token_budget(), Some(500));
        assert_eq!(ThinkingLevel::Custom(0).token_budget(), Some(0));
    }

    #[test]
    fn test_from_str_named_levels() {
        assert_eq!(ThinkingLevel::from_str("off").unwrap(), ThinkingLevel::Off);
        assert_eq!(ThinkingLevel::from_str("OFF").unwrap(), ThinkingLevel::Off);
        assert_eq!(ThinkingLevel::from_str("Low").unwrap(), ThinkingLevel::Low);
        assert_eq!(ThinkingLevel::from_str("medium").unwrap(), ThinkingLevel::Medium);
        assert_eq!(ThinkingLevel::from_str("HIGH").unwrap(), ThinkingLevel::High);
    }

    #[test]
    fn test_from_str_custom_integer() {
        assert_eq!(ThinkingLevel::from_str("2048").unwrap(), ThinkingLevel::Custom(2048));
        assert_eq!(ThinkingLevel::from_str("0").unwrap(), ThinkingLevel::Custom(0));
    }

    #[test]
    fn test_from_str_invalid() {
        assert!(ThinkingLevel::from_str("bad").is_err());
        assert!(ThinkingLevel::from_str("-1").is_err());
        assert!(ThinkingLevel::from_str("").is_err());
    }

    #[test]
    fn test_default_is_off() {
        assert_eq!(ThinkingLevel::default(), ThinkingLevel::Off);
    }

    #[test]
    fn test_serde_roundtrip() {
        let levels = [
            ThinkingLevel::Off,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::Custom(4096),
        ];
        for level in &levels {
            let serialized = serde_yaml::to_string(level).unwrap();
            let deserialized: ThinkingLevel = serde_yaml::from_str(&serialized).unwrap();
            assert_eq!(&deserialized, level);
        }
    }
}
