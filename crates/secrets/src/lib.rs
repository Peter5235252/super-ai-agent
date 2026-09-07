//! OS-backed secret storage (Windows Credential Manager) for BYOK keys.
//!
//! API keys are written to the OS credential store, never to SQLite,
//! config files, logs or the model prompt. Values are held in
//! [`secrecy::SecretString`] so accidental Debug/log exposure is
//! prevented and memory is zeroized on drop.
#![forbid(unsafe_code)]

use keyring::Entry;
use secrecy::{ExposeSecret, SecretString};
use thiserror::Error;

/// Well-known service name for all Super-AI credentials.
pub const SERVICE_NAME: &str = "super-ai";

#[derive(Debug, Error)]
pub enum SecretError {
    #[error("credential store error: {0}")]
    Store(#[from] keyring::Error),
    #[error("no secret stored for '{account}'")]
    Missing { account: String },
}

pub type Result<T> = std::result::Result<T, SecretError>;

/// Stateless wrapper over the OS credential store. Credentials are keyed
/// by `service` + `account`; Super-AI uses `service = "super-ai"` and
/// `account = provider name`.
pub struct SecretStore;

impl SecretStore {
    pub fn set(service: &str, account: &str, secret: &SecretString) -> Result<()> {
        let entry = Entry::new(service, account)?;
        entry.set_password(secret.expose_secret())?;
        Ok(())
    }

    pub fn get(service: &str, account: &str) -> Result<Option<SecretString>> {
        let entry = Entry::new(service, account)?;
        match entry.get_password() {
            Ok(password) => Ok(Some(SecretString::new(password.into()))),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn delete(service: &str, account: &str) -> Result<()> {
        let entry = Entry::new(service, account)?;
        entry.delete_credential()?;
        Ok(())
    }

    pub fn require(service: &str, account: &str) -> Result<SecretString> {
        match Self::get(service, account)? {
            Some(secret) => Ok(secret),
            None => Err(SecretError::Missing {
                account: account.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round-trip against the real OS store. Tagged so it can be skipped in
    // headless CI if the credential service is unavailable.
    #[test]
    fn roundtrip_smoke() {
        let account = format!("test-account-{}", std::process::id());
        let secret = SecretString::new("sk-test-value".to_string().into());
        SecretStore::set(SERVICE_NAME, &account, &secret).unwrap();
        let read = SecretStore::get(SERVICE_NAME, &account).unwrap().unwrap();
        assert_eq!(read.expose_secret(), "sk-test-value");
        SecretStore::delete(SERVICE_NAME, &account).unwrap();
        assert!(SecretStore::get(SERVICE_NAME, &account).unwrap().is_none());
    }
}
