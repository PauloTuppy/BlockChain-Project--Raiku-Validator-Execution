use std::sync::Arc;

use crossbeam_queue::SegQueue;
use solana_sdk::transaction::SanitizedTransaction;
// For get_message_hash and potentially constructing dummy transactions if needed later.
// Note: Constructing a valid SanitizedTransaction for tests can be complex.
use solana_sdk::message::Message; 
use solana_sdk::hash::Hash; // For dummy message hash

use wasmtime::{
    Caller, Engine, Extern, Instance, Linker, Module, Store, Trap, AsContextMut
};

// Placeholder for a real WASM module.
// For testing, you could create a WAT file and compile it to WASM.
// Example WAT for basic arithmetic, logging, and memory export:
// (module
//   (func $add (param $a i32) (param $b i32) (result i32)
//     local.get $a
//     local.get $b
//     i32.add
//   )
//   (export "add" (func $add))
//
//   ;; Import for host logging
//   (func $host_log_impl (import "env" "host_log") (param i32 i32))
//
//   ;; Function to demonstrate logging
//   (func $log_message_from_wasm
//     ;; Assume string "Hello from WASM!" is at memory offset 0
//     (data (i32.const 0) "Hello from WASM!")
//     i32.const 0    ;; memory offset of the string
//     i32.const 16   ;; length of the string "Hello from WASM!"
//     call $host_log_impl
//   )
//   (export "log_message_from_wasm" (func $log_message_from_wasm))
//
//   ;; Function for conceptual fee calculation
//   (func $calculate_fee (param $instruction_count i32) (result i64)
//     local.get $instruction_count
//     i32.const 100 ;; some base fee per instruction
//     i32.mul
//     i64.extend_i32_s ;; convert to i64
//   )
//   (export "calculate_fee" (func $calculate_fee))
//
//   (memory 1) ;; Declare a memory, min 1 page (64KiB). Essential for most operations.
//   (export "memory" (memory 0))
// )

use wasmtime::{
    Config, Engine, Module, OptLevel
};

// Placeholder for a real WASM module.
// For testing, you could create a WAT file and compile it to WASM.
// Example WAT for basic arithmetic, logging, and memory export:
// (module
//   (func $add (param $a i32) (param $b i32) (result i32)
//     local.get $a
//     local.get $b
//     i32.add
//   )
//   (export "add" (func $add))
//
//   ;; Import for host logging
//   (func $host_log_impl (import "env" "host_log") (param i32 i32))
//
//   ;; Function to demonstrate logging
//   (func $log_message_from_wasm
//     ;; Assume string "Hello from WASM!" is at memory offset 0
//     (data (i32.const 0) "Hello from WASM!")
//     i32.const 0    ;; memory offset of the string
//     i32.const 16   ;; length of the string "Hello from WASM!"
//     call $host_log_impl
//   )
//   (export "log_message_from_wasm" (func $log_message_from_wasm))
//
//   ;; Function for conceptual fee calculation
//   (func $calculate_fee (param $instruction_count i32) (result i64)
//     local.get $instruction_count
//     i32.const 100 ;; some base fee per instruction
//     i32.mul
//     i64.extend_i32_s ;; convert to i64
//   )
//   (export "calculate_fee" (func $calculate_fee))
//
//   (memory 1) ;; Declare a memory, min 1 page (64KiB). Essential for most operations.
//   (export "memory" (memory 0))
// )

pub struct ExecutionNode {
    engine: Engine,
    module: Module, // The pre-compiled WASM module
    tx_queue: Arc<SegQueue<SanitizedTransaction>>,
    // In a fuller version, you'd have:
    // state_cache: Arc<RwLock<YourStateType>>,
}

impl ExecutionNode {
    pub fn new(wasm_bytes: &[u8]) -> Self {
        let mut config = Config::new();
        config.parallel_compilation(true);
        // OptLevel::Speed is a common choice for good runtime performance.
        // OptLevel::SpeedAndSize might offer more but with longer compile times.
        config.cranelift_opt_level(OptLevel::Speed);
        // You can also enable other features like SIMD if your WASM modules use it:
        // config.wasm_simd(true);
        // For fuel consumption (to prevent runaway executions):
        // config.consume_fuel(true);

        let engine = Engine::new(&config).expect("Failed to create Wasmtime engine with custom config");

        let module = Module::from_binary(&engine, wasm_bytes)
            .expect("Failed to compile WASM module from binary");

        Self {
            engine,
            module,
            tx_queue: Arc::new(SegQueue::new()),
        }
    }

    // Method to add a transaction to the queue for processing
    pub fn submit_transaction(&self, tx: SanitizedTransaction) {
        self.tx_queue.push(tx);
        println!("Transaction {:?} submitted to queue.", tx.get_message_hash());
    }

    // Main transaction processing loop as per your `async fn process(&self)`
    pub async fn process(&self) {
        println!("ExecutionNode process loop started.");

        // --- Linker Setup for Host Function Imports ---
        // The Linker allows defining host functions that WASM modules can import.
        // This is crucial for "Bind Solana SDK functions to WASM imports".
        let mut linker = Linker::new(&self.engine);

        // Example host function: `host_log`
        // WASM module would import this as: (import "env" "host_log" (func ...))
        linker.func_wrap(
            "env", // Corresponds to the first part of WASM import e.g., (import "env" "host_log" ...)
            "host_log", // Corresponds to the second part of WASM import
            |mut caller: Caller<'_, ()>, ptr: u32, len: u32| { // Host state is `()`
                let memory = match caller.get_export("memory") {
                    Some(Extern::Memory(mem)) => mem,
                    _ => return Err(Trap::new("WASM module must export 'memory' to use host_log")),
                };
                // Read the string from WASM memory
                let data = memory.data(&caller)
                    .get(ptr as usize..(ptr + len) as usize)
                    .ok_or_else(|| Trap::new("host_log: pointer/length out of bounds"))?;
                
                match std::str::from_utf8(data) {
                    Ok(s) => {
                        println!("[WASM Log]: {}", s);
                    }
                    Err(_) => return Err(Trap::new("host_log: invalid UTF-8 string from WASM")),
                }
                Ok(())
            }
        ).expect("Failed to define host_log function in linker");
        // You would add more host functions here for Solana SDK interactions, state access, etc.

        loop {
            if let Some(sanitized_tx) = self.tx_queue.pop() {
                println!("Processing transaction with message hash: {:?}", sanitized_tx.get_message_hash());

                // Create a new store for each transaction. Host state is `()`.
                let mut store = Store::new(&self.engine, ());
                // To limit execution, you can add fuel to the store:
                // store.add_fuel(1_000_000).expect("Failed to add fuel");

                let instance = match linker.instantiate_async(&mut store, &self.module).await {
                    Ok(inst) => inst,
                    Err(e) => {
                        eprintln!(
                            "Failed to instantiate WASM module for tx {:?}: {}",
                            sanitized_tx.get_message_hash(), e
                        );
                        continue; // Skip this transaction
                    }
                };

                // --- WASM Runtime Feature: Basic arithmetic operations (PoC) ---
                // Assumes WASM module exports: (export "add" (func $add (param i32 i32) (result i32)))
                match instance.get_typed_func::<(i32, i32), i32, _>(&mut store, "add").await {
                    Ok(add_func) => {
                        match add_func.call_async(&mut store, (15, 27)).await {
                            Ok(result) => println!("WASM 'add(15, 27)' result: {}", result),
                            Err(Trap::FuelExhausted) => eprintln!("WASM 'add' call ran out of fuel."),
                            Err(e) => eprintln!("Error calling 'add' in WASM: {}", e),
                        }
                    }
                    Err(_) => { /* Function might not exist, or wrong signature. Log if necessary. */ }
                }
                
                // --- WASM Runtime Feature: Transaction fee calculation (First Use Case) ---
                // Assumes WASM exports: (export "calculate_fee" (func $calculate_fee (param i32) (result i64)))
                // The parameter could be instruction count, data size, etc.
                let instruction_count = sanitized_tx.message().instructions().len() as i32;
                match instance.get_typed_func::<i32, i64, _>(&mut store, "calculate_fee").await {
                    Ok(fee_func) => {
                        match fee_func.call_async(&mut store, instruction_count).await {
                            Ok(fee) => println!("WASM calculated fee for {} instructions: {}", instruction_count, fee),
                            Err(e) => eprintln!("Error calling 'calculate_fee' in WASM: {}", e),
                        }
                    }
                    Err(_) => { /* Optional: log if 'calculate_fee' is not found */ }
                }

                // --- WASM Runtime Feature: Memory Sandboxing via Wasmtime’s Memory API ---
                // Wasmtime provides memory sandboxing by default. Each instance has its own linear memory.
                // Host can interact with it if WASM exports its memory (usually named "memory").
                if let Some(Extern::Memory(memory)) = instance.get_export(store.as_context_mut(), "memory") {
                    // Example: Write data to WASM memory (e.g., transaction data)
                    let tx_bytes_example = b"sample_tx_data";
                    if memory.data_size(&store) > tx_bytes_example.len() { // Check if memory is large enough
                        // memory.write(&mut store, 0, tx_bytes_example).expect("Failed to write to WASM memory");
                        // println!("Conceptually wrote tx data to WASM memory at offset 0.");
                    }
                    // Reading data would be similar: memory.read(&store, offset, &mut buffer)?;
                } else {
                    // This would be unusual if your WASM needs to interact with complex data.
                    // println!("WASM module does not export 'memory'.");
                }

                // Example of calling a WASM function that uses an imported host function
                match instance.get_typed_func::<(), (), _>(&mut store, "log_message_from_wasm").await {
                    Ok(log_fn) => {
                        if let Err(e) = log_fn.call_async(&mut store, ()).await {
                            eprintln!("Error calling 'log_message_from_wasm': {}", e);
                        }
                    }
                    Err(_) => { /* 'log_message_from_wasm' not found */ }
                }
                
                // Check remaining fuel if it was enabled
                // if let Some(fuel) = store.get_fuel()? { println!("Remaining fuel: {}", fuel); }

                println!("Finished processing transaction {:?}.", sanitized_tx.get_message_hash());
            } else {
                // Queue is empty, wait a bit before checking again.
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
        }
    }
}

// Helper to create a dummy SanitizedTransaction for testing.
// This is simplified; real transactions are more complex.
fn create_dummy_sanitized_transaction(id: u8) -> SanitizedTransaction {
    use solana_sdk::{
        message::MessageHeader,
        pubkey::Pubkey,
        transaction::Transaction,
        sanitize::Sanitize,
    };
    // Create a minimal message
    let payer = Pubkey::new_unique(); // Dummy payer
    let instructions = vec![]; // No actual instructions for this dummy
    let message = Message::new_with_blockhash(
        &instructions,
        Some(&payer),
        &Hash::new_unique(), // Dummy blockhash
    );
    let tx = Transaction::new_unsigned(message);
    // Normally, you'd sign it here: tx.sign(&[&keypair], recent_blockhash);
    // For SanitizedTransaction, it needs to be sanitized.
    // This is a simplified way to get a SanitizedTransaction for testing,
    // but it might not be fully representative of a network transaction.
    // The `SanitizedTransaction::try_from_legacy_transaction` or similar might be needed.
    // For now, we'll use a test utility if available or a simplified path.
    // The `solana-transaction-status` crate has `SanitizedTransaction::from_transaction_for_tests`
    // but that's for tests. Let's assume a simplified path for now.
    // This is a placeholder, as direct construction is involved.
    // A more robust way is to parse a serialized, signed transaction.
    // For the purpose of this example, we'll use a simplified approach.
    let mut tx_to_sanitize = tx;
    tx_to_sanitize.sanitize().expect("Failed to sanitize dummy transaction"); // Perform basic sanitization
    
    // The `SanitizedTransaction` type is often an enum or a wrapper.
    // `solana_sdk::transaction::SanitizedTransaction` is actually a type alias for
    // `solana_transaction_status::SanitizedTransaction`.
    // Let's assume we can construct it from a sanitized `Transaction`.
    // This part is tricky without pulling in more specific test utilities or fully signing.
    // For the purpose of queueing, we'll assume this is valid enough.
    // In a real test, you'd use `solana-program-test` or similar.
    
    // This is a conceptual conversion. The actual API might differ.
    // `SanitizedTransaction` is often created internally by the runtime after parsing and verifying.
    // For this example, we'll focus on the `ExecutionNode`'s ability to *receive* it.
    // We'll use a trick: create a versioned message and then use a test constructor if possible.
    // If not, we'll have to simplify further or comment out submission.
    
    // Let's use a known way to create a SanitizedTransaction for testing from solana_ledger
    // This is a bit of a hack for this standalone example.
    use solana_sdk::transaction::VersionedTransaction;
    let versioned_tx = VersionedTransaction::from(tx_to_sanitize);
    SanitizedTransaction::try_from_versioned_transaction(versioned_tx).unwrap()
}


#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("Execution Node Main: Starting...");

    // --- Load WASM Module Bytes ---
    // This WAT string defines a simple WASM module with 'add', 'calculate_fee',
    // 'log_message_from_wasm' functions and exports its memory.
    // It also imports 'host_log' from the "env" module.
    let wat_bytes = r#"
        (module
          (func $add (param $a i32) (param $b i32) (result i32)
            local.get $a
            local.get $b
            i32.add
          )
          (export "add" (func $add))

          (func $host_log_impl (import "env" "host_log") (param i32 i32))
          (func $log_message_from_wasm
            (data (i32.const 0) "Hello from WASM!") ;; String in memory
            i32.const 0    ;; pointer to string
            i32.const 16   ;; length of string
            call $host_log_impl
          )
          (export "log_message_from_wasm" (func $log_message_from_wasm))

          (func $calculate_fee (param $instruction_count i32) (result i64)
            local.get $instruction_count
            i32.const 50 ;; Lamports per instruction (example)
            i32.mul
            i64.extend_i32_s
          )
          (export "calculate_fee" (func $calculate_fee))

          (memory $mem 1) ;; Declare memory, 1 page (64KB)
          (export "memory" (memory $mem))
        )
    "#;
    let wasm_bytes = wat::parse_str(wat_bytes).expect("Failed to parse WAT to WASM bytes");
    // Alternatively, load from a .wasm file:
    // let wasm_bytes = std::fs::read("path/to/your_module.wasm")?;

    // --- Create and Start ExecutionNode ---
    let execution_node = Arc::new(ExecutionNode::new(&wasm_bytes));
    println!("Execution Node instance created.");

    let processing_node_handle = Arc::clone(&execution_node);
    tokio::spawn(async move {
        processing_node_handle.process().await; // This loop runs indefinitely
    });
    println!("ExecutionNode process loop spawned.");

    // --- Simulate Submitting Transactions (Optional for Testing) ---
    // Creating valid SanitizedTransactions for testing can be involved.
    // Here's a conceptual way to submit a few dummy transactions.
    // Note: `create_dummy_sanitized_transaction` is a simplified helper.
    // for i in 0..3 {
    //     // The dummy transaction creation might need adjustment based on SDK specifics
    //     // For now, this part is illustrative.
    //     // let dummy_tx = create_dummy_sanitized_transaction(i);
    //     // execution_node.submit_transaction(dummy_tx);
    //     // tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    // }
    println!("(Skipping dummy transaction submission in this example to simplify SDK dependencies for SanitizedTransaction creation)");


    // Keep the main task alive for a demonstration period.
    // In a real server, this would run indefinitely or wait for a shutdown signal.
    println!("Main: Simulating work for 5 seconds...");
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    println!("Execution Node Main: Shutting down simulation.");
    Ok(())
}
use anyhow::Result; // For Result type
use rustls::{Certificate, PrivateKey}; // For certs

// Dummy function to represent loading certs and key
fn load_certs_and_key() -> Result<(Vec<Certificate>, PrivateKey)> {
    // In a real app, load these from files or a secure store
    // For example purposes, these would be rcgen-generated or actual certs
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let cert_der = cert.serialize_der().unwrap();
    let key_der = cert.serialize_private_key_der();
    let key = rustls::PrivateKey(key_der);
    let cert_chain = vec![rustls::Certificate(cert_der)];
    Ok((cert_chain, key))
}

async fn setup_quic_server() -> Result<quinn::Endpoint, Box<dyn std::error::Error>> {
    let (cert_chain, key) = load_certs_and_key()?; // Helper function to load certs

    let mut server_crypto = rustls::ServerConfig::builder()
        .with_safe_defaults()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)?;

    let mut server_config = quinn::ServerConfig::with_crypto(Arc::new(server_crypto));
    
    let mut transport_config = quinn::TransportConfig::default();
    transport_config.keep_alive_interval(Some(std::time::Duration::from_secs(10)));
    // Other transport settings like `max_idle_timeout` can also be relevant.
    transport_config.max_idle_timeout(
        Some(tokio::time::Duration::from_secs(30).try_into().map_err(|e| format!("Invalid duration: {}", e))?)
    );


    server_config.transport = Arc::new(transport_config);
    
    let listen_addr: std::net::SocketAddr = "0.0.0.0:4433".parse()?;
    let endpoint = quinn::Endpoint::server(server_config, listen_addr)?;
    Ok(endpoint)
}