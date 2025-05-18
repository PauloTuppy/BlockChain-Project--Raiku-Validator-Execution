use solana_sdk::transaction::SanitizedTransaction;
use tract_onnx::prelude::*;
use tract_ndarray; // tract_onnx::prelude usually brings in ndarray via tract_ndarray
use std::path::Path;
use std::fs::File;
use std::io::Read;
use serde::{Serialize, Deserialize}; // For FeatureScaler serialization

// Define NodeID type alias
pub type NodeID = i64; // Assuming node IDs are represented as i64, matching model output

// Define the number of features. This MUST match the Python script.
// FEATURES = [
//     'tx_size', 'num_instructions', 'fee_priority',
//     'account_access_pattern', 'program_id', 'timestamp', 'network_congestion'
// ]
const FEATURE_COUNT: usize = 7; // Based on the FEATURES list in Python

// Placeholder for FeatureScaler.
// In a real scenario, this would store scaling parameters (e.g., mean, std)
// learned from the training data (e.g., using sklearn.preprocessing.StandardScaler).
#[derive(Serialize, Deserialize, Debug)]
pub struct FeatureScaler {
    // Example: if using standardization
    mean: Vec<f32>,
    std_dev: Vec<f32>,
    // Add other parameters as needed depending on the scaling method
}

impl FeatureScaler {
    // Load scaler parameters from a file (e.g., "scaler.bin")
    // This file would be created by the Python script after fitting the scaler.
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let scaler: FeatureScaler = bincode::deserialize(&buffer)?;
        Ok(scaler)
    }

    // Apply the transformation to the features
    // The input `features` vector should have FEATURE_COUNT elements.
    pub fn transform(&self, mut features: Vec<f32>) -> Vec<f32> {
        if features.len() != self.mean.len() || features.len() != self.std_dev.len() {
            // Handle error: feature count mismatch
            // For simplicity, returning original features or panicking.
            // In production, proper error handling is needed.
            eprintln!("Feature count mismatch in FeatureScaler::transform. Expected {}, got {}.", self.mean.len(), features.len());
            return features; 
        }

        for i in 0..features.len() {
            if self.std_dev[i] != 0.0 { // Avoid division by zero
                features[i] = (features[i] - self.mean[i]) / self.std_dev[i];
            } else {
                features[i] = features[i] - self.mean[i]; // Or handle as appropriate
            }
        }
        features
    }

    // Dummy method to create a scaler for placeholder purposes if scaler.bin doesn't exist
    #[allow(dead_code)]
    fn dummy() -> Self {
        Self {
            mean: vec![0.5; FEATURE_COUNT], // Dummy mean
            std_dev: vec![0.1; FEATURE_COUNT], // Dummy std_dev
        }
    }
}

pub struct Scheduler {
    model: TypedModel, // Changed from tract_onnx::InferenceModel for clarity with TypedModel
    feature_normalizer: FeatureScaler,
}

impl Scheduler {
    pub fn new(model_path: &str, scaler_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading ONNX model from: {}", model_path);
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .into_optimized()?
            .into_runnable()?; // This creates a TypedModel

        println!("Loading feature scaler from: {}", scaler_path);
        let feature_normalizer = FeatureScaler::load(scaler_path)
            .or_else(|e| {
                eprintln!("Warning: Failed to load scaler from '{}': {:?}. Using dummy scaler.", scaler_path, e);
                Ok(FeatureScaler::dummy()) // Fallback to dummy for testing if load fails
            })?;


        Ok(Self {
            model,
            feature_normalizer,
        })
    }

    // Extracts features from a SanitizedTransaction.
    // This needs to mirror the feature extraction logic in your Python script.
    fn extract_features(&self, tx: &SanitizedTransaction) -> Vec<f32> {
        // Placeholder implementation.
        // You need to implement the logic to get:
        // 'tx_size', 'num_instructions', 'fee_priority',
        // 'account_access_pattern', 'program_id', 'timestamp', 'network_congestion'
        // from the SanitizedTransaction object.

        let tx_size = tx.message_data().len() as f32;
        let num_instructions = tx.message().instructions().len() as f32;
        
        // Fee priority: This is complex. Solana transactions have compute unit limits and fees.
        // You might need to parse specific instructions (e.g., ComputeBudgetInstruction::SetComputeUnitPrice)
        // or use a default/average if not explicitly set.
        let fee_priority = 0.0f32; // Placeholder

        // Account access pattern: Could be number of unique accounts, ratio of read-only to writable, etc.
        let account_access_pattern = tx.message().account_keys().len() as f32; // Placeholder: just num accounts

        // Program ID: This needs careful handling. A transaction can invoke multiple programs.
        // You might focus on the first instruction's program ID or create a categorical feature.
        // For now, a placeholder. If using it as a categorical feature, it needs encoding.
        let program_id_feature = 0.0f32; // Placeholder

        // Timestamp: Solana transactions don't inherently carry a user-set timestamp for this purpose.
        // You'd likely use the arrival time at the scheduler. For now, a placeholder.
        let timestamp_feature = 0.0f32; // Placeholder: e.g., SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f32();

        // Network congestion: This would be an external input to the scheduler,
        // representing the current state of the network or execution nodes.
        let network_congestion_feature = 0.0f32; // Placeholder

        vec![
            tx_size,
            num_instructions,
            fee_priority,
            account_access_pattern,
            program_id_feature,
            timestamp_feature,
            network_congestion_feature,
        ]
    }

    pub fn predict(&self, tx: &SanitizedTransaction) -> Result<NodeID, Box<dyn std::error::Error>> {
        let features = self.extract_features(tx);
        if features.len() != FEATURE_COUNT {
            return Err(format!("Extracted feature count mismatch. Expected {}, got {}.", FEATURE_COUNT, features.len()).into());
        }

        let normalized_features = self.feature_normalizer.transform(features);
        
        // Convert features to Tract's Tensor type
        let input_tensor = tract_ndarray::Array1::from_vec(normalized_features)
            .into_shape((1, FEATURE_COUNT))?; // Batch size 1
        
        // Run inference
        // The tvec! macro creates a TVec<TensorValue>
        let result_tensors = self.model.run(tvec!(input_tensor.into()))?;

        // Assuming the model outputs a tensor with a single i64 value (the predicted node ID)
        // and it's the first output tensor.
        // The actual output name and structure depend on your scikit-learn model and ONNX conversion.
        // For GradientBoostingClassifier, the output is typically class labels.
        let prediction_value = result_tensors[0]
            .to_array_view::<i64>()? // GBC outputs i64 labels
            .get_item(()) // Get the single value from a 0-d array (scalar)
            .cloned()
            .ok_or("Failed to extract scalar prediction from model output")?;
            
        Ok(prediction_value)
    }
}

// Example usage (e.g., in a test or main function)
#[cfg(test)]
mod tests {
    use super::*;
    use solana_sdk::{
        message::Message,
        pubkey::Pubkey,
        signature::{Keypair, Signer, Signature},
        hash::Hash,
    };

    fn create_dummy_sanitized_transaction() -> SanitizedTransaction {
        let payer = Keypair::new();
        let dummy_instruction = solana_sdk::instruction::Instruction::new_with_bincode(
            Pubkey::new_unique(), // Program ID
            &0u8, // Instruction data
            vec![], // No accounts
        );
        let message = Message::new_with_blockhash(
            &[dummy_instruction],
            Some(&payer.pubkey()),
            &Hash::new_unique(), // Dummy blockhash
        );
        // Need to sign the transaction to make it SanitizedTransaction::try_new compatible
        let tx = solana_sdk::transaction::Transaction::new(&[&payer], message, Hash::new_unique());
        SanitizedTransaction::try_from_legacy_transaction(tx).unwrap()
    }
    
    // Helper to create a dummy scaler.bin for testing
    fn create_dummy_scaler_file(path: &str) {
        let scaler = FeatureScaler {
            mean: vec![0.5; FEATURE_COUNT],
            std_dev: vec![0.1; FEATURE_COUNT],
        };
        let encoded: Vec<u8> = bincode::serialize(&scaler).unwrap();
        std::fs::write(path, encoded).unwrap();
    }


    #[test]
    fn test_scheduler_prediction() {
        // Ensure you have a "scheduler_model.onnx" from your Python script
        // and a "scaler.bin" (or use the dummy creation).
        let model_path = "../scheduler_model.onnx"; // Adjust path as needed
        let scaler_path = "test_scaler.bin";
        create_dummy_scaler_file(scaler_path);


        // Check if the ONNX model file exists before trying to load it
        if !Path::new(model_path).exists() {
            eprintln!("ONNX model file not found at: {}. Skipping test_scheduler_prediction.", model_path);
            eprintln!("Please generate 'scheduler_model.onnx' using the Python script.");
            return; // Or panic, or handle as appropriate for your test setup
        }


        match Scheduler::new(model_path, scaler_path) {
            Ok(scheduler) => {
                let dummy_tx = create_dummy_sanitized_transaction();
                match scheduler.predict(&dummy_tx) {
                    Ok(node_id) => {
                        println!("Predicted Node ID: {}", node_id);
                        // Add assertions here based on expected behavior
                        assert!(node_id >= 0); // Example: node ID should be non-negative
                    }
                    Err(e) => panic!("Prediction failed: {:?}", e),
                }
            }
            Err(e) => panic!("Scheduler initialization failed: {:?}", e),
        }
        std::fs::remove_file(scaler_path).unwrap(); // Clean up dummy scaler
    }
}