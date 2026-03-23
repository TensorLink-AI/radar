#!/bin/bash
# =============================================================================
# EvoLoop — Create Test Wallets
#
# Creates miner + validator wallets for testing. On localnet, funds them from
# the Alice dev account and registers them on the subnet. On testnet, creates
# the wallets and reminds you to fund via faucet.
#
# All wallets use no password for ease of automated testing.
#
# Usage:
#   bash scripts/create_test_wallets.sh                         # 3 miners, 2 validators, localnet
#   bash scripts/create_test_wallets.sh --miners 5              # 5 miners
#   bash scripts/create_test_wallets.sh --validators 3          # 3 validators
#   bash scripts/create_test_wallets.sh --network test          # testnet (creates owner + subnet + all wallets)
#   bash scripts/create_test_wallets.sh --network test --skip-subnet --netuid 123  # join existing subnet
#   bash scripts/create_test_wallets.sh --stake 500             # stake 500 TAO per validator
#   bash scripts/create_test_wallets.sh --fund 50000            # fund 50k TAO per wallet (localnet)
#   bash scripts/create_test_wallets.sh --skip-register         # create wallets only, don't register
#   bash scripts/create_test_wallets.sh --skip-subnet           # don't create subnet (already exists)
#   bash scripts/create_test_wallets.sh --list                  # list existing wallets and exit
#
# On localnet: funds everything from Alice (pre-funded dev account).
# On testnet:  creates owner wallet, registers subnet, creates all wallets.
#              You must fund wallets via faucet BEFORE running (script checks balances).
# =============================================================================

set -euo pipefail

# ── Defaults ──
NUM_MINERS=3
NUM_VALIDATORS=2
NETWORK=local
NETUID=1
STAKE_AMOUNT=1000
FUND_AMOUNT=100000
SKIP_REGISTER=false
SKIP_SUBNET=false
LIST_ONLY=false

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --miners)        NUM_MINERS="$2"; shift 2 ;;
        --validators)    NUM_VALIDATORS="$2"; shift 2 ;;
        --network)       NETWORK="$2"; shift 2 ;;
        --netuid)        NETUID="$2"; shift 2 ;;
        --stake)         STAKE_AMOUNT="$2"; shift 2 ;;
        --fund)          FUND_AMOUNT="$2"; shift 2 ;;
        --skip-register) SKIP_REGISTER=true; shift ;;
        --skip-subnet)   SKIP_SUBNET=true; shift ;;
        --list)          LIST_ONLY=true; shift ;;
        --help|-h)
            head -18 "$0" | tail -14
            exit 0 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── List mode ──
if [ "$LIST_ONLY" = true ]; then
    python3 << 'PYEOF'
import bittensor as bt
import os

wallet_dir = os.path.expanduser("~/.bittensor/wallets")
if not os.path.isdir(wallet_dir):
    print("No wallets found.")
    exit(0)

wallets = sorted(os.listdir(wallet_dir))
print(f"Found {len(wallets)} wallet(s) in {wallet_dir}:\n")
for name in wallets:
    try:
        w = bt.Wallet(name=name)
        cold = w.coldkeypub.ss58_address
        hot = w.hotkey.ss58_address
        print(f"  {name}")
        print(f"    coldkey: {cold}")
        print(f"    hotkey:  {hot}")
    except Exception:
        print(f"  {name} (incomplete — missing coldkey or hotkey)")
    print()
PYEOF
    exit 0
fi

# ── Banner ──
NETUID_DISPLAY="$NETUID"
if [ "$SKIP_SUBNET" = false ] && [ "$NETWORK" != "local" ] && [ "$NETUID" = "1" ]; then
    NETUID_DISPLAY="auto (new subnet)"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         EvoLoop — Create Test Wallets                        ║"
echo "║                                                              ║"
printf "║  Network:    %-48s ║\n" "$NETWORK"
printf "║  Netuid:     %-48s ║\n" "$NETUID_DISPLAY"
printf "║  Owner:      %-48s ║\n" "owner (creates subnet)"
printf "║  Miners:     %-48s ║\n" "$NUM_MINERS"
printf "║  Validators: %-48s ║\n" "$NUM_VALIDATORS"
printf "║  Fund:       %-48s ║\n" "${FUND_AMOUNT} TAO each (localnet only)"
printf "║  Stake:      %-48s ║\n" "${STAKE_AMOUNT} TAO per validator"
printf "║  Password:   %-48s ║\n" "none (no password)"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ── Main ──
python3 << PYEOF
import bittensor as bt
import sys

network = '${NETWORK}'
netuid = ${NETUID}
num_miners = ${NUM_MINERS}
num_validators = ${NUM_VALIDATORS}
fund_amount = ${FUND_AMOUNT}
stake_amount = ${STAKE_AMOUNT}
skip_register = '${SKIP_REGISTER}' == 'true'
skip_subnet = '${SKIP_SUBNET}' == 'true'

sub = bt.Subtensor(network=network)
print(f'Connected to {network} — block {sub.block}')

# ── Funding source (localnet only) ──
alice = None
if network == 'local':
    alice = bt.Wallet(name='alice-dev')
    alice.create_coldkey_from_uri('//Alice', use_password=False, overwrite=True, suppress=True)
    alice.create_new_hotkey(use_password=False, overwrite=True, suppress=True)
    bal = sub.get_balance(alice.coldkeypub.ss58_address)
    print(f'Alice balance: {bal}')
    if bal < 10000:
        print('ERROR: Alice not pre-funded. Is subtensor running with devnet-ready image?')
        sys.exit(1)

def fund_if_local(w):
    """Fund wallet from Alice on localnet. On testnet, check balance and warn."""
    cold = w.coldkeypub.ss58_address
    bal = sub.get_balance(cold)
    if alice and bal < 10000:
        sub.transfer(wallet=alice, destination_ss58=cold,
                     amount=bt.Balance.from_tao(fund_amount),
                     wait_for_inclusion=True, wait_for_finalization=True)
    elif not alice and bal < 100:
        print(f'  WARNING: {w.name} has low balance ({bal} TAO).')
        print(f'           Fund it: btcli wallet faucet --wallet.name {w.name} --subtensor.network {network}')

# ── Owner wallet + subnet ──
print(f'\n--- Owner ---')
owner = bt.Wallet(name='owner')
owner.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)
owner_cold = owner.coldkeypub.ss58_address
owner_hot = owner.hotkey.ss58_address
print(f'  owner           coldkey={owner_cold[:16]}...  hotkey={owner_hot[:16]}...')
fund_if_local(owner)

if not skip_subnet and not skip_register:
    owner_bal = sub.get_balance(owner_cold)
    if owner_bal < 100:
        print(f'\n  ERROR: Owner wallet has insufficient balance ({owner_bal} TAO).')
        if network != 'local':
            print(f'  Fund it first: btcli wallet faucet --wallet.name owner --subtensor.network {network}')
        sys.exit(1)

    try:
        result = sub.register_subnet(wallet=owner, mev_protection=False)
        # On success, the chain assigns the next available netuid
        # Try to detect it from the result or scan for it
        if hasattr(result, 'netuid'):
            netuid = result.netuid
        else:
            # Scan subnets to find the one owned by this wallet
            for nid in range(1, 100):
                try:
                    info = sub.get_subnet_info(nid)
                    if info and hasattr(info, 'owner_ss58') and info.owner_ss58 == owner_cold:
                        netuid = nid
                        break
                except Exception:
                    break
        print(f'  Subnet created — netuid {netuid}')
    except Exception as e:
        err = str(e)
        if 'already' in err.lower() or 'exist' in err.lower():
            print(f'  Subnet already exists on netuid {netuid}')
        else:
            print(f'  Subnet registration: {e}')
            print(f'  If subnet already exists, use --skip-subnet --netuid <N>')
elif skip_subnet:
    print(f'  Skipping subnet creation (--skip-subnet), using netuid {netuid}')

print(f'\n  Using netuid: {netuid}')

# ── Helper ──
def create_wallet(name, role):
    """Create a wallet with no password, optionally fund and register."""
    w = bt.Wallet(name=name)
    w.create_if_non_existent(coldkey_use_password=False, hotkey_use_password=False)

    cold = w.coldkeypub.ss58_address
    hot = w.hotkey.ss58_address

    fund_if_local(w)
    bal = sub.get_balance(cold)

    # Register
    if not skip_register:
        if bal < 10:
            print(f'  WARNING: {name} balance too low to register ({bal} TAO). Skipping registration.')
            print(f'  {name:15s}  coldkey={cold[:16]}...  hotkey={hot[:16]}...  balance={bal}  [created, NOT registered]')
            return w
        try:
            sub.burned_register(wallet=w, netuid=netuid, mev_protection=False)
        except Exception:
            pass  # Already registered

    # Stake validators
    if role == 'validator' and not skip_register:
        try:
            sub.add_stake(wallet=w, netuid=netuid, hotkey_ss58=hot,
                          amount=bt.Balance.from_tao(stake_amount), mev_protection=False)
        except Exception:
            pass

    final_bal = sub.get_balance(cold)
    status = 'registered + staked' if role == 'validator' else 'registered'
    if skip_register:
        status = 'created'
    print(f'  {name:15s}  coldkey={cold[:16]}...  hotkey={hot[:16]}...  balance={final_bal}  [{status}]')
    return w

# ── Create wallets ──
print(f'\n--- Validators ({num_validators}) ---')
for i in range(num_validators):
    create_wallet(f'validator{i}', 'validator')

print(f'\n--- Miners ({num_miners}) ---')
for i in range(num_miners):
    create_wallet(f'miner{i}', 'miner')

# ── Summary ──
print(f'\n--- Summary ---')
total = 1 + num_validators + num_miners  # owner + validators + miners
print(f'  Wallets created: {total} (1 owner + {num_validators} validators + {num_miners} miners)')
print(f'  Network: {network}, netuid: {netuid}')

if not skip_register:
    meta = sub.metagraph(netuid=netuid)
    print(f'  Metagraph: {meta.n} neurons on netuid {netuid}')
    for uid in range(meta.n):
        hk = meta.hotkeys[uid][:16] if uid < len(meta.hotkeys) else '?'
        print(f'    UID {uid}: hotkey={hk}... stake={meta.S[uid]:.0f}')

if network != 'local':
    unfunded = []
    for i in range(num_validators):
        w = bt.Wallet(name=f'validator{i}')
        if sub.get_balance(w.coldkeypub.ss58_address) < 100:
            unfunded.append(f'validator{i}')
    for i in range(num_miners):
        w = bt.Wallet(name=f'miner{i}')
        if sub.get_balance(w.coldkeypub.ss58_address) < 100:
            unfunded.append(f'miner{i}')
    if unfunded:
        print(f'\n  Wallets needing funding ({len(unfunded)}):')
        for name in unfunded:
            print(f'    btcli wallet faucet --wallet.name {name} --subtensor.network {network}')
    else:
        print(f'\n  All wallets funded.')

print(f'\nDone. Next steps:')
if not skip_register:
    print(f'  bash scripts/multi_node_test.sh --skip-subtensor --netuid {netuid}')
else:
    print(f'  Re-run without --skip-register to register wallets on the subnet')
PYEOF

if [ $? -eq 0 ]; then
    ok "All wallets created"
    echo ""
    info "Wallet files stored in ~/.bittensor/wallets/"
    info "List them anytime: bash scripts/create_test_wallets.sh --list"
else
    fail "Wallet creation failed"
    exit 1
fi
