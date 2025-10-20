from enum import Enum


class Lang(str, Enum):
    EN = "EN"
    ZH_SIM = "ZH_SIM"   # Simplified Chinese (CN)
    ZH_TR = "ZH_TR"   # Traditional Chinese (HK)

def display_lang(lang: Lang) -> str:
    return {"EN": "English", "ZH_SIM": "简体中文", "ZH_TR": "繁體中文"}[lang]

if __name__ == "__main__":
    target_lang = Lang.ZH_TR
    print(display_lang(target_lang))