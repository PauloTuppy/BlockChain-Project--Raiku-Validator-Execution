use solana_sdk::transaction::SanitizedTransaction;
use std::net::SocketAddr;
use quinn::{ClientConfig, Connection, Endpoint}; // Ensure these are imported
use std::sync::Arc; // Ensure Arc is imported
use anyhow::Result; // For Result type

// DEV ONLY: For skipping server certificate verification (unsafe for production)
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self(Arc::new(rustls::crypto::ring::default_provider())))
    }
}

impl rustls::client::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::Certificate,
        _intermediates: &[rustls::Certificate],
        _server_name: &rustls::ServerName,
        _scts: &mut dyn Iterator<Item = &[u8]>,
        _ocsp_response: &[u8],
        _now: std::time::SystemTime,
    ) -> Result<rustls::client::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::ServerCertVerified::assertion())
    }

    fn request_scts(&self) -> bool { true }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.0.cipher_suites[0].resolve().unwrap().supported_sig_schemes().to_vec()
    }
}

pub struct ValidatorSidecar {
    exec_node_address: SocketAddr,
    connection: Connection,
}


impl ValidatorSidecar {
    pub async fn new(target_exec_node: SocketAddr) -> Result<Self> {
        let client_crypto = rustls::ClientConfig::builder()
            .with_safe_defaults()
            .with_custom_certificate_verifier(SkipServerVerification::new()) // DEV ONLY
            .with_no_client_auth();
        
        let mut client_config = quinn::ClientConfig::new(Arc::new(client_crypto));

        // Client-side transport config can also be set if needed,
        // though keep-alive is often driven by the server.
        let mut transport_config = quinn::TransportConfig::default();
        // Example: Client can also request a keep-alive, though server usually dictates.
        // transport_config.keep_alive_interval(Some(std::time::Duration::from_secs(10)));
        transport_config.max_idle_timeout(
             Some(tokio::time::Duration::from_secs(30).try_into().map_err(|e| format!("Invalid duration: {}", e))?)
        );
        client_config.transport_config(Arc::new(transport_config));


        let mut endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
        endpoint.set_default_client_config(client_config);

        println!("[Sidecar] Connecting to QUIC server at {}...", target_exec_node);
        let conn = endpoint
            .connect(target_exec_node, "localhost")? // "localhost" is server_name for SNI
            .await?;
        println!("[Sidecar] QUIC connection established to {}", target_exec_node);

        Ok(ValidatorSidecar {
            exec_node_address: target_exec_node,
            connection: conn,
        })
    }

    // Routes a transaction to the connected execution node via a new QUIC stream.
    pub async fn route_tx(&self, tx: &SanitizedTransaction) -> Result<()> {
        let tx_data = bincode::serialize(tx)
            .map_err(|e| anyhow::anyhow!("Failed to serialize transaction: {}", e))?;

        // Open a new bidirectional stream
        let (mut send_stream, mut recv_stream) = self.connection.open_bi().await?;
        println!("[Sidecar] Opened QUIC stream to send tx {:?}", tx.get_message_hash());

        // Send the transaction data
        send_stream.write_all(&tx_data).await?;
        send_stream.finish().await?; // Gracefully close the send side of the stream

        println!("[Sidecar] Transaction data sent. Waiting for ack/response...");

        // Optionally, wait for an acknowledgment or result from the execution node
        let mut response_buffer = Vec::new();
        recv_stream.read_to_end(&mut response_buffer).await?; // Max size can be limited
        
        if response_buffer == b"ACK" { // Example ACK
            println!("[Sidecar] Received ACK for tx {:?}", tx.get_message_hash());
        } else {
            println!("[Sidecar] Received response for tx {:?}: {:?}", tx.get_message_hash(), String::from_utf8_lossy(&response_buffer));
        }

        Ok(())
    }

    // Example of how the sidecar might receive transactions (e.g., from a local queue or RPC)
    // This is highly dependent on how the sidecar integrates with the validator.
    pub async fn transaction_ingestion_loop(&self /* source: SomeTxSource */) {
        // In a real scenario, this loop would get transactions from the validator
        // or a proxy. For now, let's simulate it.
        // loop {
        //     let tx: SanitizedTransaction = source.get_next_transaction().await;
        //     if let Err(e) = self.route_tx(&tx).await {
        //         eprintln!("[Sidecar] Failed to route transaction: {}", e);
        //     }
        // }
        println!("[Sidecar] Transaction ingestion loop would run here.");
    }
}

// Example main function for the sidecar
/*
#[tokio::main]
async fn main() -> Result<()> {
    // Replace with the actual address of your running ExecutionNode QUIC server
    let exec_node_addr: SocketAddr = "127.0.0.1:4433".parse()?;

    let sidecar = ValidatorSidecar::new(exec_node_addr).await?;
    println!("[Sidecar] ValidatorSidecar initialized, connected to {}", exec_node_addr);

    // --- Simulate sending a transaction ---
    // In a real scenario, SanitizedTransaction would come from the validator.
    // Creating a valid SanitizedTransaction here is complex.
    // This is a placeholder for where you'd get transactions.
    // For example:
    // let dummy_tx = create_dummy_sanitized_transaction_for_sidecar();
    // sidecar.route_tx(&dummy_tx).await?;

    // sidecar.transaction_ingestion_loop().await; // Start the loop if you have a tx source

    // Keep the main task alive for demonstration
    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
    Ok(())
}

// Helper to create a dummy SanitizedTransaction for testing the sidecar.
// This would be similar to the one in your execution_node/src/main.rs
fn create_dummy_sanitized_transaction_for_sidecar() -> SanitizedTransaction {
    use solana_sdk::{
        message::Message,
        pubkey::Pubkey,
        transaction::Transaction,
        hash::Hash,
        signature::{Keypair, Signer},
    };
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
    let tx = Transaction::new(&[&payer], message, Hash::new_unique());
    SanitizedTransaction::try_from_legacy_transaction(tx).unwrap()
}
*/