"""Debug script: read a miner's chain commitment raw."""
import sys
import json
import bittensor as bt

netuid = 279
hotkey = "5CkBnGFvMbrfGGFzZAFrCoR7yfG9vDiDHgsJtxLwSi5mviEN"

print(f"Querying commitment for {hotkey[:16]}... on netuid {netuid}")
sub = bt.Subtensor(network="test")
substrate = sub.substrate

# Method 1: direct query
print("\n--- substrate.query ---")
try:
    result = substrate.query(
        module="Commitments",
        storage_function="CommitmentOf",
        params=[netuid, hotkey],
    )
    print(f"result type: {type(result)}")
    print(f"result: {result}")
    if result:
        print(f"result.value type: {type(result.value)}")
        print(f"result.value: {json.dumps(result.value, indent=2, default=str)}")
except Exception as e:
    print(f"ERROR: {e}")

# Method 2: get_commitment (may fail)
print("\n--- subtensor.get_commitment ---")
try:
    raw = sub.get_commitment(netuid=netuid, uid=86)
    print(f"raw: {raw}")
except Exception as e:
    print(f"ERROR: {e}")

# Method 3: RPC request
print("\n--- rpc_request ---")
try:
    # Try querying the storage key directly
    from substrateinterface import StorageKey
    key = substrate.create_storage_key(
        "Commitments", "CommitmentOf", [netuid, hotkey],
    )
    result = substrate.rpc_request("state_getStorage", [key.to_hex()])
    print(f"hex result: {result}")
    if result and result.get("result"):
        raw_bytes = bytes.fromhex(result["result"][2:])
        print(f"raw bytes len: {len(raw_bytes)}")
        # Try to find JSON in the raw bytes
        for i in range(len(raw_bytes)):
            if raw_bytes[i:i+1] == b'{':
                try:
                    text = raw_bytes[i:].decode("utf-8")
                    end = text.index('}') + 1
                    print(f"Found JSON at offset {i}: {text[:end]}")
                except Exception:
                    pass
except Exception as e:
    print(f"ERROR: {e}")
