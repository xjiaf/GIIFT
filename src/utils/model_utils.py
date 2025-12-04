import torch
import logging

logger = logging.getLogger("__main__")


def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


def get_lang_code(lang: str, backbone: str) -> str:
    """
    Return the correct language code for MBART or NLLB, given:
      - lang: two-letter language key, e.g. 'en', 'fr', 'zh', ...
      - backbone: 'mbart' or 'nllb'.
    """

    MBART_LANG_MAP = {
        'ar': 'ar_AR',
        'cs': 'cs_CZ',
        'de': 'de_DE',
        'en': 'en_XX',
        'es': 'es_XX',
        'et': 'et_EE',
        'fi': 'fi_FI',
        'fr': 'fr_XX',
        'gu': 'gu_IN',
        'hi': 'hi_IN',
        'it': 'it_IT',
        'ja': 'ja_XX',
        'kk': 'kk_KZ',
        'ko': 'ko_KR',
        'lt': 'lt_LT',
        'lv': 'lv_LV',
        'my': 'my_MM',
        'ne': 'ne_NP',
        'nl': 'nl_XX',
        'ro': 'ro_RO',
        'ru': 'ru_RU',
        'si': 'si_LK',
        'tr': 'tr_TR',
        'vi': 'vi_VN',
        'zh': 'zh_CN',
        'af': 'af_ZA',
        'uk': 'uk_UA',
    }

    NLLB_LANG_MAP = {
        'ar': 'arb_Arab',
        'cs': 'ces_Latn',
        'de': 'deu_Latn',
        'en': 'eng_Latn',
        'es': 'spa_Latn',
        'et': 'est_Latn',
        'fi': 'fin_Latn',
        'fr': 'fra_Latn',
        'gu': 'guj_Gujr',
        'hi': 'hin_Deva',
        'it': 'ita_Latn',
        'ja': 'jpn_Jpan',
        'kk': 'kaz_Cyrl',
        'ko': 'kor_Hang',
        'lt': 'lit_Latn',
        'lv': 'lav_Latn',
        'my': 'mya_Mymr',
        'ne': 'npi_Deva',
        'nl': 'nld_Latn',
        'ro': 'ron_Latn',
        'ru': 'rus_Cyrl',
        'si': 'sin_Sinh',
        'tr': 'tur_Latn',
        'vi': 'vie_Latn',
        'zh': 'zho_Hans',
        'af': 'afr_Latn',
        'uk': 'ukr_Cyrl',
    }

    # Select the appropriate code map
    if backbone == 'mbart':
        if lang not in MBART_LANG_MAP:
            raise ValueError(f"No MBART language code mapping for '{lang}'")
        return MBART_LANG_MAP[lang]

    elif backbone == 'nllb':
        if lang not in NLLB_LANG_MAP:
            raise ValueError(f"No NLLB language code mapping for '{lang}'")
        return NLLB_LANG_MAP[lang]

    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Must be 'mbart' or 'nllb'.")


def send_to_cuda(batch):
    if torch.is_tensor(batch):
        batch = batch.cuda()
    else:
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].cuda()
    return batch


class EarlyStopMonitor(object):

    def __init__(self, max_round=3, higher_better=True, tolerance=1e-5):
        self.max_round = max_round
        self.num_bad_rounds = 0
        self.best_score = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_score):
        if self.best_score is None:
            self.best_score = curr_score
            logger.info(f"Initial score set to: {self.best_score}")
            return False  # Do not stop

        if self.higher_better:
            improvement = curr_score - self.best_score
        else:
            improvement = self.best_score - curr_score

        if improvement > self.tolerance:
            previous_best = self.best_score
            self.best_score = curr_score
            self.num_bad_rounds = 0
            logger.info(f"New best score: {self.best_score} (Improved)\n"
                        f"Previous best score: {previous_best}")
        else:
            self.num_bad_rounds += 1
            logger.info(f"Current score: {curr_score} (No improvement)\n"
                        f"The best score: {self.best_score}\n"
                        f"Bad rounds: {self.num_bad_rounds}/{self.max_round}")

        # Check if we need to stop
        stop = self.num_bad_rounds >= self.max_round
        if stop:
            logger.info("Early stopping triggered.")
        return stop
