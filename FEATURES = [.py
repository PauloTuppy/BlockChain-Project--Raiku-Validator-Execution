FEATURES = [
    'tx_size',                # The size of the transaction in bytes.
    'num_instructions',       # The number of instructions in the transaction.
    'fee_priority',           # Priority fee paid by the transaction (e.g., lamports per compute unit).
    'account_access_pattern', # A representation of which accounts are read/written.
                              # This might need further definition (e.g., number of accounts, hot accounts involved).
    'program_id',             # The main program ID being invoked.
    'timestamp',              # The timestamp when the transaction was observed in the mempool.
    'network_congestion'      # A metric representing current network load (e.g., recent TPS, queue depth).
]