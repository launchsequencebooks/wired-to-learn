"""Banking77 for ticket triage: teams, a word tokenizer, and the data splits."""
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wtlkit import load_banking77

TEAMS = {  # which support team handles each of the 77 intents
    "Cards": ["activate_my_card", "card_about_to_expire", "card_arrival", "card_delivery_estimate", "card_linking",
              "card_not_working", "card_swallowed", "change_pin", "contactless_not_working", "get_disposable_virtual_card",
              "get_physical_card", "getting_spare_card", "getting_virtual_card", "order_physical_card", "pin_blocked",
              "virtual_card_not_working", "visa_or_mastercard", "disposable_card_limits", "card_acceptance",
              "apple_pay_or_google_pay", "supported_cards_and_currencies", "atm_support"],
    "Fraud & security": ["card_payment_not_recognised", "cash_withdrawal_not_recognised", "compromised_card",
                         "direct_debit_payment_not_recognised", "lost_or_stolen_card", "lost_or_stolen_phone"],
    "Payments & disputes": ["card_payment_fee_charged", "card_payment_wrong_exchange_rate", "declined_card_payment",
                            "pending_card_payment", "reverted_card_payment?", "request_refund", "Refund_not_showing_up",
                            "extra_charge_on_statement", "transaction_charged_twice", "cash_withdrawal_charge",
                            "declined_cash_withdrawal", "pending_cash_withdrawal", "wrong_amount_of_cash_received",
                            "wrong_exchange_rate_for_cash_withdrawal"],
    "Transfers": ["balance_not_updated_after_bank_transfer", "beneficiary_not_allowed", "cancel_transfer",
                  "declined_transfer", "failed_transfer", "pending_transfer", "receiving_money", "transfer_fee_charged",
                  "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing"],
    "Top-ups & exchange": ["automatic_top_up", "balance_not_updated_after_cheque_or_cash_deposit", "pending_top_up",
                           "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
                           "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card", "verify_top_up",
                           "exchange_charge", "exchange_rate", "exchange_via_app", "fiat_currency_support"],
    "Identity & account": ["age_limit", "country_support", "edit_personal_details", "passcode_forgotten",
                           "terminate_account", "unable_to_verify_identity", "verify_my_identity",
                           "verify_source_of_funds", "why_verify_identity"],
}
MAX_LEN = 48                                              # words per ticket (longer ones are cut)

(train_texts, train_y), (test_texts, test_y), INTENTS = load_banking77()
TEAM_NAMES = list(TEAMS)
INTENT_TO_TEAM = np.array([next(i for i, t in enumerate(TEAMS.values()) if name in t) for name in INTENTS])
assert sum(len(v) for v in TEAMS.values()) == len(INTENTS) == 77

rng = np.random.default_rng(0)                            # hold out 10% of training tickets for validation
perm = rng.permutation(len(train_texts))
val_idx, tr_idx = perm[:1000], perm[1000:]


def words(text):
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower())


counts = Counter(w for i in tr_idx for w in words(train_texts[i]))
VOCAB = ["<pad>", "<unk>"] + sorted(w for w, c in counts.items() if c >= 2)   # rare words become <unk>
WORD_ID = {w: i for i, w in enumerate(VOCAB)}


def encode(texts):
    """Texts -> (N, MAX_LEN) word ids, padded with 0."""
    out = torch.zeros(len(texts), MAX_LEN, dtype=torch.long)
    for n, t in enumerate(texts):
        ids = [WORD_ID.get(w, 1) for w in words(t)][:MAX_LEN]
        out[n, :len(ids)] = torch.tensor(ids)
    return out


def split(name):
    """('texts', intent labels as a tensor) for 'train', 'val' or 'test'."""
    if name == "test":
        return test_texts, torch.tensor(test_y)
    idx = tr_idx if name == "train" else val_idx
    return [train_texts[i] for i in idx], torch.tensor(train_y[idx])
