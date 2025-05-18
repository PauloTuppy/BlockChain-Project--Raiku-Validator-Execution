// Potentially using Anchor-like macros if you build or integrate such a framework
// For now, let's use serde for serialization as a general example.
// If Anchor is a direct dependency:
// use anchor_lang::prelude::*;
use serde::{Serialize, Deserialize};

/// Trait defining the core capabilities of a custom execution environment.
pub trait CustomEnvironmentLogic {
    type State; // Environment-specific state
    type Error; // Environment-specific error type
    type CallArgs; // Arguments for calling into the environment

    /// Initializes the environment with given parameters.
    fn init(params: &[u8]) -> Result<Self::State, Self::Error>
    where
        Self: Sized;

    /// Processes a call within the environment.
    fn process_call(state: &mut Self::State, args: Self::CallArgs) -> Result<Vec<u8>, Self::Error>;

    // Potentially other common functions: query, state_hash, etc.
}

/// Example: A simple DeFi environment template.
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct DeFiEnvironmentState {
    pub total_value_locked: u64,
    pub token_balances: std::collections::HashMap<String, u64>, // Token symbol -> balance
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DeFiCallArgs {
    Deposit { token: String, amount: u64 },
    Withdraw { token: String, amount: u64 },
    QueryBalance { token: String },
}

#[derive(Debug)]
pub enum DeFiError {
    TokenNotFound,
    InsufficientBalance,
    InvalidArguments,
    InitializationFailed,
}

pub struct DeFiEnvironment;

impl CustomEnvironmentLogic for DeFiEnvironment {
    type State = DeFiEnvironmentState;
    type Error = DeFiError;
    type CallArgs = DeFiCallArgs;

    fn init(_params: &[u8]) -> Result<Self::State, Self::Error> {
        // Deserialize params if needed for initialization
        Ok(DeFiEnvironmentState::default())
    }

    fn process_call(state: &mut Self::State, args: Self::CallArgs) -> Result<Vec<u8>, Self::Error> {
        match args {
            DeFiCallArgs::Deposit { token, amount } => {
                let balance = state.token_balances.entry(token.clone()).or_insert(0);
                *balance += amount;
                state.total_value_locked += amount; // Simplified TVL update
                println!("[DeFiEnv] Deposited {} of {}", amount, token);
                Ok(b"DepositSuccessful".to_vec())
            }
            DeFiCallArgs::Withdraw { token, amount } => {
                let balance = state.token_balances.entry(token.clone()).or_insert(0);
                if *balance < amount {
                    return Err(DeFiError::InsufficientBalance);
                }
                *balance -= amount;
                state.total_value_locked -= amount; // Simplified TVL update
                println!("[DeFiEnv] Withdrew {} of {}", amount, token);
                Ok(b"WithdrawSuccessful".to_vec())
            }
            DeFiCallArgs::QueryBalance { token } => {
                let balance = state.token_balances.get(&token).unwrap_or(&0);
                println!("[DeFiEnv] Queried balance for {}: {}", token, balance);
                Ok(balance.to_le_bytes().to_vec())
            }
        }
    }
}

// Similar templates could be created for NFTEnvironment, AIEnvironment, etc.
// Each would define its own State, CallArgs, and Error types, and implement
// the CustomEnvironmentLogic trait.