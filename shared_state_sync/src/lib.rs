use serde::{Serialize, Deserialize};

// Common error type for state sync operations
#[derive(Debug)]
pub enum StateSyncError {
    DeserializationError(String),
    SerializationError(String),
    NetworkError(String),
    InvalidStateTransition,
}

/// Represents a minimal, serializable account state.
/// This structure would need to align with Firedancer's expectations
/// or have a clear mapping to/from it.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AccountState {
    pub lamports: u64,
    pub data: Vec<u8>,
    pub owner_program_id: [u8; 32], // e.g., a public key
    pub executable: bool,
    pub rent_epoch: u64,
}

/// Represents a delta or update to the state.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum StateUpdate {
    AccountWrite {
        account_id: [u8; 32], // Public key of the account
        new_state: AccountState,
    },
    AccountDelete {
        account_id: [u8; 32],
    },
    // Other types of updates: e.g., slot advancement, bank hash
}

/// Trait for a component that can provide state updates.
pub trait StateProvider {
    /// Fetches state updates since a given point (e.g., slot or hash).
    fn get_updates_since(&self, last_known_slot: u64) -> Result<Vec<StateUpdate>, StateSyncError>;
}

/// Trait for a component that can apply state updates.
pub trait StateApplier {
    /// Applies a batch of state updates.
    fn apply_updates(&mut self, updates: Vec<StateUpdate>) -> Result<(), StateSyncError>;
}

// Implementations of these traits would be specific to your blockchain's
// state management and Firedancer's architecture.
// For Firedancer compatibility, you might need FFI (Foreign Function Interface)
// if Firedancer exposes C APIs for state ingestion, or if it can load Rust libraries.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_account_state_serialization() {
        let account = AccountState {
            lamports: 1000,
            data: vec![1, 2, 3],
            owner_program_id: [1u8; 32],
            executable: false,
            rent_epoch: 10,
        };
        let serialized = bincode::serialize(&account).unwrap();
        let deserialized: AccountState = bincode::deserialize(&serialized).unwrap();
        assert_eq!(account, deserialized);
    }
}