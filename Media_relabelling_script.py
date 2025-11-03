import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import matplotlib
import re

# ===== Load INSURANCE file =====
data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\media_data_global.csv")
# ===================== BEGIN MAPPING BLOCK =====================
import pandas as pd
import re
from io import StringIO
from difflib import get_close_matches

# ---- Paste ONLY the three needed columns (exactly as in your table) ----
_ref_text = r"""Company Name	Primary Industry	Headquarters - Country/Region
Comcast Corporation (NasdaqGS:CMCSA)	Cable and Satellite	United States
Charter Communications, Inc. (NasdaqGS:CHTR)	Cable and Satellite	United States
The Trade Desk, Inc. (NasdaqGM:TTD)	Advertising	United States
Publicis Groupe S.A. (ENXTPA:PUB)	Advertising	France
Fox Corporation (NasdaqGS:FOXA)	Broadcasting	United States
EchoStar Corporation (NasdaqGS:SATS)	Cable and Satellite	United States
Paramount Skydance Corporation (NasdaqGS:PSKY)	Broadcasting	United States
Informa plc (LSE:INF)	Advertising	United Kingdom
News Corporation (NasdaqGS:NWSA)	Publishing	United States
Omnicom Group Inc. (NYSE:OMC)	Advertising	United States
Focus Media Information Technology Co., Ltd. (SZSE:002027)	Advertising	China
China Satellite Communications Co., Ltd. (SHSE:601698)	Cable and Satellite	China
The Interpublic Group of Companies, Inc. (NYSE:IPG)	Advertising	United States
The New York Times Company (NYSE:NYT)	Publishing	United States
Liberty Broadband Corporation (NasdaqGS:LBRD.K)	Cable and Satellite	United States
Sirius XM Holdings Inc. (NasdaqGS:SIRI)	Cable and Satellite	United States
RTL Group S.A. (XTRA:RRTL)	Broadcasting	Luxembourg
Nippon Television Holdings, Inc. (TSE:9404)	Broadcasting	Japan
Nexstar Media Group, Inc. (NasdaqGS:NXST)	Broadcasting	United States
TBS Holdings,Inc. (TSE:9401)	Broadcasting	Japan
Dentsu Group Inc. (TSE:4324)	Advertising	Japan
Springer Nature AG & Co. KGaA (XTRA:SPG)	Publishing	Germany
CyberAgent, Inc. (TSE:4751)	Advertising	Japan
WPP plc (LSE:WPP)	Advertising	United Kingdom
China Literature Limited (SEHK:772)	Publishing	China
Fuji Media Holdings, Inc. (TSE:4676)	Broadcasting	Japan
Leo Group Co., Ltd. (SZSE:002131)	Advertising	China
Oriental Pearl Group Co.,Ltd. (SHSE:600637)	Cable and Satellite	China
PT Elang Mahkota Teknologi Tbk (IDX:EMTK)	Broadcasting	Indonesia
NIQ Global Intelligence plc (NYSE:NIQ)	Advertising	United States
JCDecaux SE (ENXTPA:DEC)	Advertising	France
Jiangsu Phoenix Publishing & Media Corporation Limited (SHSE:601928)	Publishing	China
Saudi Research and Media Group (SASE:4210)	Publishing	Saudi Arabia
ITV plc (LSE:ITV)	Broadcasting	United Kingdom
Mobvista Inc. (SEHK:1860)	Advertising	Singapore
Kadokawa Corporation (TSE:9468)	Publishing	Japan
SES S.A. (BDL:SESGL)	Cable and Satellite	Luxembourg
TEGNA Inc. (NYSE:TGNA)	Broadcasting	United States
Lagardere SA (ENXTPA:MMB)	Publishing	France
Canal+ SA (LSE:CAN)	Broadcasting	France
BlueFocus Intelligent Communications Group Co., Ltd. (SZSE:300058)	Advertising	China
China South Publishing & Media Group Co., Ltd (SHSE:601098)	Publishing	China
MBC Group (SASE:4072)	Broadcasting	Saudi Arabia
MultiChoice Group Limited (JSE:MCG)	Cable and Satellite	South Africa
Affle 3i Limited (NSEI:AFFLE)	Advertising	India
People.cn CO., LTD (SHSE:603000)	Publishing	China
Magnite, Inc. (NasdaqGS:MGNI)	Advertising	United States
MFE-Mediaforeurope N.V. (BIT:MFEB)	Broadcasting	Italy
SKY Perfect JSAT Holdings Inc. (TSE:9412)	Cable and Satellite	Japan
COL Group Co.,Ltd. (SZSE:300364)	Publishing	China
Hakuhodo DY Holdings Inc (TSE:2433)	Advertising	Japan
TX Group AG (SWX:TXGN)	Publishing	Switzerland
Ströer SE & Co. KGaA (XTRA:SAX)	Advertising	Germany
Zhejiang Publishing & Media Co., Ltd. (SHSE:601921)	Publishing	China
Jiangsu Broadcasting Cable Information Network Corporation Limited (SHSE:600959)	Broadcasting	China
Sun TV Network Limited (NSEI:SUNTV)	Broadcasting	India
Megacable Holdings, S. A. B. de C. V. (BMV:MEGA CPO)	Cable and Satellite	Mexico
Shandong Publishing&Media Co.,Ltd (SHSE:601019)	Publishing	China
Xinhua Winshare Publishing and Media Co., Ltd. (SEHK:811)	Publishing	China
Wasu Media Holding Co.,Ltd (SZSE:000156)	Cable and Satellite	China
Hebei Broadcasting Wireless Media Co., Ltd. (SZSE:301551)	Cable and Satellite	China
China Science Publishing & Media Ltd. (SHSE:601858)	Publishing	China
TF1 SA (ENXTPA:TFI)	Broadcasting	France
TV Asahi Holdings Corporation (TSE:9409)	Broadcasting	Japan
Sanoma Oyj (HLSE:SANOMA)	Publishing	Finland
Chinese Universe Publishing and Media Group Co., Ltd. (SHSE:600373)	Publishing	China
DoubleVerify Holdings, Inc. (NYSE:DV)	Advertising	United States
John Wiley & Sons, Inc. (NYSE:WLY)	Publishing	United States
Eutelsat Communications S.A. (ENXTPA:ETL)	Cable and Satellite	France
Guangdong Advertising Group Co.,Ltd (SZSE:002400)	Advertising	China
Easy Click Worldwide Network Technology Co., Ltd. (SZSE:301171)	Advertising	China
GMO internet, Inc. (TSE:4784)	Advertising	Japan
Zhewen Interactive Group Co., Ltd. (SHSE:600986)	Advertising	China
Xinhuanet Co., Ltd. (SHSE:603888)	Publishing	China
China Publishing & Media Holdings Co., Ltd. (SHSE:601949)	Publishing	China
Central China Land Media CO.,LTD (SZSE:000719)	Publishing	China
Métropole Télévision S.A. (ENXTPA:MMT)	Broadcasting	France
Louis Hachette Group S.A. (ENXTPA:ALHG)	Publishing	France
Hunan TV & Broadcast Intermediary Co., Ltd. (SZSE:000917)	Advertising	China
JiShi Media Co., Ltd. (SHSE:601929)	Cable and Satellite	China
Havas N.V. (ENXTAM:HAVAS)	Advertising	France
Ipsos SA (ENXTPA:IPS)	Advertising	France
Integral Ad Science Holding Corp. (NasdaqGS:IAS)	Advertising	United States
Guizhou BC&TV Information Network CO.,LTD (SHSE:600996)	Cable and Satellite	China
Southern Publishing and Media Co.,Ltd. (SHSE:601900)	Publishing	China
Arabian Contracting Services Company (SASE:4071)	Advertising	Saudi Arabia
Beijing Gehua Catv Network Co.,Ltd. (SHSE:600037)	Cable and Satellite	China
Guangdong South New Media Co.,Ltd. (SZSE:300770)	Broadcasting	China
ProSiebenSat.1 Media SE (XTRA:PSM)	Broadcasting	Germany
Changjiang Publishing & Media Co.,Ltd (SHSE:600757)	Publishing	China
Alma Media Oyj (HLSE:ALMA)	Publishing	Finland
CITIC Guoan Information Industry Co., Ltd. (SZSE:000839)	Cable and Satellite	China
Guangdong Guangzhou Daily Media Co., Ltd. (SZSE:002181)	Publishing	China
Cheil Worldwide Inc. (KOSE:A030000)	Advertising	South Korea
Inmyshow Digital Technology(Group)Co.,Ltd. (SHSE:600556)	Advertising	China
Qunabox Group Limited (SEHK:917)	Advertising	China
Newsmax Inc. (NYSE:NMAX)	Broadcasting	United States
Sunwave Communications Co.Ltd (SZSE:002115)	Advertising	China
Atresmedia Corporación de Medios de Comunicación, S.A. (BME:A3M)	Broadcasting	Spain
MNTN, Inc. (NYSE:MNTN)	Advertising	United States
Stagwell Inc. (NasdaqGS:STGW)	Advertising	United States
PT Surya Citra Media Tbk (IDX:SCMA)	Broadcasting	Indonesia
Grupo Televisa, S.A.B. (BMV:TLEVISA CPO)	Cable and Satellite	Mexico
Nine Entertainment Co. Holdings Limited (ASX:NEC)	Broadcasting	Australia
4imprint Group plc (LSE:FOUR)	Advertising	United Kingdom
NanJi E-Commerce Co., LTD (SZSE:002127)	Advertising	China
Zee Entertainment Enterprises Limited (NSEI:ZEEL)	Broadcasting	India
Xiamen Jihong Co., Ltd (SZSE:002803)	Advertising	China
Criteo S.A. (NasdaqGS:CRTO)	Advertising	France
Shenzhen Topway Video Communication Co., Ltd (SZSE:002238)	Cable and Satellite	China
Altice USA, Inc. (NYSE:ATUS)	Cable and Satellite	United States
PT Solusi Sinergi Digital Tbk (IDX:WIFI)	Advertising	Indonesia
Shanghai Xinhua Media Co., Ltd. (SHSE:600825)	Publishing	China
Clear Channel Outdoor Holdings, Inc. (NYSE:CCO)	Advertising	United States
Hangzhou Onechance Tech Crop. (SZSE:300792)	Advertising	China
Hubei Radio & Television Information Network Co., Ltd. (SZSE:000665)	Broadcasting	China
Ibotta, Inc. (NYSE:IBTA)	Advertising	United States
Sinclair, Inc. (NasdaqGS:SBGI)	Broadcasting	United States
Time Publishing and Media Co., Ltd. (SHSE:600551)	Publishing	China
Emerald Holding, Inc. (NYSE:EEX)	Advertising	United States
Huawen Media Group (SZSE:000793)	Publishing	China
Cable One, Inc. (NYSE:CABO)	Cable and Satellite	United States
Three's Company Media Group Co., Ltd. (SHSE:605168)	Advertising	China
Guangdong Yowant Technology Group Co., Ltd. (SZSE:002291)	Advertising	China
Guangdong Insight Brand Marketing Group Co.,Ltd. (SZSE:300781)	Advertising	China
Heilongjiang Publishing & Media Co., Ltd. (SHSE:605577)	Publishing	China
Guangxi Radio and Television Information Network Corporation Limited (SHSE:600936)	Cable and Satellite	China
Network18 Media & Investments Limited (NSEI:NETWORK18)	Broadcasting	India
Future plc (LSE:FUTR)	Publishing	United Kingdom
VGI Public Company Limited (SET:VGI)	Advertising	Thailand
TV TOKYO Holdings Corporation (TSE:9413)	Broadcasting	Japan
APG|SGA SA (SWX:APGN)	Advertising	Switzerland
Citic Press Corporation (SZSE:300788)	Publishing	China
Chengdu B-ray Media Co.,Ltd. (SHSE:600880)	Publishing	China
Shanghai Fengyuzhu Culture Technology Co., Ltd. (SHSE:603466)	Advertising	China
Scholastic Corporation (NasdaqGS:SCHL)	Publishing	United States
NRJ Group SA (ENXTPA:NRG)	Broadcasting	France
Guangdong Brandmax Marketing Co.,Ltd. (SZSE:300805)	Advertising	China
Vertoz Limited (NSEI:VERTOZ)	Advertising	India
Ai Robotics Inc. (TSE:247A)	Advertising	Japan
Beijing Yuanlong Yato Culture Dissemination Co.,Ltd. (SZSE:002878)	Advertising	China
Fiera Milano S.p.A. (BIT:FM)	Advertising	Italy
Mega-info Media Co.,Ltd. (SZSE:301102)	Advertising	China
Arnoldo Mondadori Editore S.p.A. (BIT:MN)	Publishing	Italy
GUOMAI Culture & Media Co., Ltd. (SZSE:301052)	Publishing	China
Storytel AB (publ) (OM:STORY B)	Publishing	Sweden
Inly Media Co., Ltd. (SHSE:603598)	Advertising	China
Qingdao Citymedia Co,. Ltd. (SHSE:600229)	Publishing	China
RCS MediaGroup S.p.A. (BIT:RCS)	Publishing	Italy
Zhejiang Huamei Holding CO., LTD. (SZSE:000607)	Advertising	China
Cablevisión Holding S.A. (BASE:CVH)	Cable and Satellite	Argentina
Plan B Media Public Company Limited (SET:PLANB)	Advertising	Thailand
Zhejiang Meorient Commerce Exhibition Inc. (SZSE:300795)	Advertising	China
Promotora de Informaciones, S.A. (BME:PRS)	Publishing	Spain
FS Development Investment Holdings (SZSE:300071)	Advertising	China
Northern United Publishing & Media (Group) Company Limited (SHSE:601999)	Publishing	China
Verve Group SE (XTRA:M8G)	Advertising	Sweden
Septeni Holdings Co., Ltd. (TSE:4293)	Advertising	Japan
Bloomsbury Publishing Plc (LSE:BMY)	Publishing	United Kingdom
Stingray Group Inc. (TSX:RAY.A)	Broadcasting	Canada
Thryv Holdings, Inc. (NasdaqCM:THRY)	Advertising	United States
Dook Media Group Limited (SZSE:301025)	Publishing	China
Nexxen International Ltd. (NasdaqGM:NEXN)	Advertising	Israel
DuZhe Publish&Media Co.,Ltd (SHSE:603999)	Publishing	China
Gannett Co., Inc. (NYSE:GCI)	Publishing	United States
Tangel Culture Co., Ltd. (SZSE:300148)	Publishing	China
Next 15 Group plc (AIM:NFG)	Advertising	United Kingdom
D. B. Corp Limited (NSEI:DBCORP)	Publishing	India
Astro-century Education&Technology Co.,Ltd (SZSE:300654)	Publishing	China
Gray Media, Inc. (NYSE:GTN)	Broadcasting	United States
LZ Technology Holdings Limited (NasdaqCM:LZMH)	Advertising	China
Shaanxi Broadcast & TV Network Intermediary(Group)Co.,Ltd. (SHSE:600831)	Cable and Satellite	China
oOh!media Limited (ASX:OML)	Advertising	Australia
Advantage Solutions Inc. (NasdaqGS:ADV)	Advertising	United States
Innocean Worldwide Inc. (KOSE:A214320)	Advertising	South Korea
iHeartMedia, Inc. (NasdaqGS:IHRT)	Broadcasting	United States
Viaplay Group AB (publ) (OM:VPLAY B)	Broadcasting	Sweden
Perion Network Ltd. (NasdaqGS:PERI)	Advertising	Israel
Rongxin Education and Culture Industry Development Co., Ltd. (SZSE:301231)	Publishing	China
WideOpenWest, Inc. (NYSE:WOW)	Cable and Satellite	United States
MPS Limited (NSEI:MPSLTD)	Publishing	India
Boston Omaha Corporation (NYSE:BOC)	Advertising	United States
Beijing Quanshi World Online Network Information Co., Ltd. (SZSE:002995)	Advertising	China
Sichuan Newsnet Media (Group) Co.,Ltd. (SZSE:300987)	Publishing	China
Pico Far East Holdings Limited (SEHK:752)	Advertising	Hong Kong
INTAGE HOLDINGS Inc. (TSE:4326)	Advertising	Japan
TechTarget, Inc. (NasdaqGS:TTGT)	Advertising	United States
Simei Media Co.,Ltd. (SZSE:002712)	Advertising	China
Guangzhou Frontop Digital Creative Technology Corporation (SZSE:301313)	Advertising	China
YouGov plc (AIM:YOU)	Advertising	United Kingdom
Jiayun Technology Inc. (SZSE:300242)	Advertising	China
Alter Ego Media S.A. (ATSE:AEM)	Broadcasting	Greece
Navneet Education Limited (NSEI:NAVNETEDUL)	Publishing	India
PubMatic, Inc. (NasdaqGM:PUBM)	Advertising	United States
"""

ref = pd.read_csv(StringIO(_ref_text), sep="\t")
ref = ref.rename(columns={
    "Company Name": "ref_name",
    "Primary Industry": "ref_industry",
    "Headquarters - Country/Region": "ref_country"
})[["ref_name", "ref_industry", "ref_country"]]

# ---- Make a fast exact-lookup dict ----
REF_MAP = {r.ref_name: (r.ref_industry, r.ref_country) for _, r in ref.iterrows()}

# ---- Alias map: left is how it might appear in your data (Company_name), right is the EXACT ref_name above ----
ALIASES = {
    # Australia / NZ
    "Nine Entertainment": "Nine Entertainment Co. Holdings Limited (ASX:NEC)",
    "Nine Entertainment Co. Holdings Limited": "Nine Entertainment Co. Holdings Limited (ASX:NEC)",
    "Seven West Media": "Seven West Media",  # not in ref block; we will smart-guess below
    "Southern Cross Media": "Southern Cross Media",  # smart-guess below
    "NZME": "NZME",  # smart-guess below
    "Sky network TV": "Sky Network Television",  # smart-guess below

    # Common punctuation/encoding variants already present in ref
    "APG|SGA SA (SWX:APGN)": "APG|SGA SA (SWX:APGN)",
    "4imprint Group plc (LSE:FOUR)": "4imprint Group plc (LSE:FOUR)",
    "Ströer SE & Co. KGaA (XTRA:SAX)": "Ströer SE & Co. KGaA (XTRA:SAX)",
    "Scholastic Corporation (NasdaqGS:SCHL)": "Scholastic Corporation (NasdaqGS:SCHL)",
    "TV Asahi Holdings": "TV Asahi Holdings Corporation (TSE:9409)",
    "TF1 Group": "TF1 SA (ENXTPA:TFI)",
    "MFE-Mediaforeurope N.V. (BIT:MFEB)": "MFE-Mediaforeurope N.V. (BIT:MFEB)",

    # US bigs not in ref (we'll guess later if needed)
    "Disney": "Disney",
    "Netflix": "Netflix",
    "Amazon": "Amazon",
    "Apple": "Apple ",

    # Canada bigs
    "Rogers Communications": "Rogers Communications",
    "BCE Inc": "BCE Inc",
}

# ---- Supplemental smart guesses for companies NOT in the ref block (based on common sense + consistency) ----
# Use existing list styles for subtype names and countries.
SUPPLEMENT = {
    # Australia / NZ media landscape
    "Seven West Media": ("Broadcasting", "Australia"),
    "Southern Cross Media": ("Broadcasting", "Australia"),
    "NZME": ("Broadcasting", "New Zealand"),
    "Sky Network Television": ("Broadcasting", "New Zealand"),
    "Sky network TV": ("Broadcasting", "New Zealand"),

    # Global majors frequently in your wider list
    "Disney": ("Broadcasting", "United States"),
    "Warner Bros Discovery": ("Broadcasting", "United States"),
    "Netflix": ("Broadcasting", "United States"),
    "Amazon": ("Advertising", "United States"),
    "Apple ": ("Advertising", "United States"),
    "Apple": ("Advertising", "United States"),
    "iHeartMedia": ("Broadcasting", "United States"),
    "iHeartMedia, Inc. (NasdaqGS:IHRT)": ("Broadcasting", "United States"),

    # Canada
    "Rogers Communications": ("Cable and Satellite", "Canada"),
    "BCE Inc": ("Cable and Satellite", "Canada"),

    # Malaysia
    "Media Prima Berhad": ("Broadcasting", "Malaysia"),

    # Middle East / Africa examples already present in ref for consistency
    "MBC Group": ("Broadcasting", "Saudi Arabia"),
    "MultiChoice": ("Cable and Satellite", "South Africa"),
}

# ---- Helper: light normalization (strip, squeeze spaces) ----
def norm(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

# ---- Try to get a mapping for one name ----
ref_names = list(REF_MAP.keys())

def map_company(name: str):
    nm = norm(name)

    # 1) Direct exact to REF
    if nm in REF_MAP:
        return REF_MAP[nm]

    # 2) Alias to REF
    if nm in ALIASES and ALIASES[nm] in REF_MAP:
        return REF_MAP[ALIASES[nm]]

    # 3) Supplemental fixed guesses
    if nm in SUPPLEMENT:
        return SUPPLEMENT[nm]

    # 4) Try cleaning trailing ticker parentheses to hit a REF key
    nm_no_ticker = re.sub(r"\s*\([^)]*\)\s*$", "", nm).strip()
    if nm_no_ticker in REF_MAP:
        return REF_MAP[nm_no_ticker]
    if nm_no_ticker in SUPPLEMENT:
        return SUPPLEMENT[nm_no_ticker]

    # 5) Fuzzy match to the official REF names (very strict)
    match = get_close_matches(nm, ref_names, n=1, cutoff=0.93)
    if match:
        return REF_MAP[match[0]]

    # 6) Keyword-based heuristic (country guess from hints; subtype bias toward Broadcasting for "TV", Publishing for "Publish", else Advertising)
    lower = nm.lower()
    if "tv" in lower or "television" in lower or "media" in lower:
        subtype = "Broadcasting"
    elif "publish" in lower or "press" in lower or "times" in lower or "wiley" in lower:
        subtype = "Publishing"
    elif "satellite" in lower or "cable" in lower or "broadband" in lower or "jsat" in lower:
        subtype = "Cable and Satellite"
    else:
        subtype = "Advertising"

    # country heuristic: use likely region keywords
    if any(k in lower for k in ["china", "beijing", "shanghai", "shenzhen", "szse", "shse", "sehk", "hong kong"]):
        country = "China" if "hong kong" not in lower else "Hong Kong"
    elif any(k in lower for k in ["india", "nsei", "bse", "zeel", "suntv"]):
        country = "India"
    elif any(k in lower for k in ["japan", "tse:", "tse:9", "tse:4", "japan"]):
        country = "Japan"
    elif any(k in lower for k in ["korea", "kose"]):
        country = "South Korea"
    elif any(k in lower for k in ["saudi", "sase"]):
        country = "Saudi Arabia"
    elif any(k in lower for k in ["mexico", "bmv:"]):
        country = "Mexico"
    elif any(k in lower for k in ["sweden", "om:vplay", "storytel"]):
        country = "Sweden"
    elif any(k in lower for k in ["finland", "hlse"]):
        country = "Finland"
    elif any(k in lower for k in ["france", "enxtpa"]):
        country = "France"
    elif any(k in lower for k in ["switzerland", "swx:"]):
        country = "Switzerland"
    elif any(k in lower for k in ["italy", "bit:"]):
        country = "Italy"
    elif any(k in lower for k in ["luxembourg", "bdl:"]):
        country = "Luxembourg"
    elif any(k in lower for k in ["spain", "bme:"]):
        country = "Spain"
    elif any(k in lower for k in ["argentina", "base:"]):
        country = "Argentina"
    elif any(k in lower for k in ["australia", "asx:"]):
        country = "Australia"
    elif any(k in lower for k in ["indonesia", "idx:"]):
        country = "Indonesia"
    elif any(k in lower for k in ["thailand", "set:"]):
        country = "Thailand"
    elif any(k in lower for k in ["sa", "south africa", "jse:"]):
        country = "South Africa"
    elif any(k in lower for k in ["united kingdom", "lse:", "(lse:", "aim:"]):
        country = "United Kingdom"
    elif any(k in lower for k in ["canada", "tsx:", "tsx:", "tse:ray.a"]):
        country = "Canada"
    elif any(k in lower for k in ["new zealand", "nz", "sky network tv", "nzme"]):
        country = "New Zealand"
    else:
        country = "United States"

    return (subtype, country)

# ---- Apply mapping to your dataframe ----
mapped = data["Company_name"].apply(map_company)
data["Media_type_label"] = mapped.str[0]
data["Country_label"]   = mapped.str[1]

# ---- Hard pins for the ANZ set to avoid any future drift ----
for exact_nm in [
    "Nine Entertainment", "Nine Entertainment Co. Holdings Limited (ASX:NEC)",
    "Seven West Media", "Southern Cross Media", "NZME",
    "Sky network TV", "Sky Network Television"
]:
    mask = data["Company_name"].str.strip().str.casefold() == exact_nm.strip().casefold()
    if mask.any():
        if "Nine Entertainment" in exact_nm:
            data.loc[mask, ["Media_type_label","Country_label"]] = ("Broadcasting","Australia")
        elif exact_nm in ("NZME",):
            data.loc[mask, ["Media_type_label","Country_label"]] = ("Broadcasting","New Zealand")
        elif "Sky" in exact_nm:
            data.loc[mask, ["Media_type_label","Country_label"]] = ("Broadcasting","New Zealand")
        else:
            data.loc[mask, ["Media_type_label","Country_label"]] = ("Broadcasting","Australia")

# ---- Absolutely no NaNs: fill any residual holes with a safe default ----
data["Media_type_label"] = data["Media_type_label"].fillna("Advertising")
data["Country_label"]    = data["Country_label"].fillna("United States")

# (Optional) quick sanity check for rows we care about
# print(data.loc[data["Company_name"].str.contains("Nine|Seven West|Southern Cross|NZME|Sky", case=False, na=False),
#                ["Company_name","Media_type_label","Country_label"]].head(20))
# ===================== END MAPPING BLOCK =====================

# ---------- Write final dataset ----------
output_path = r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\Nine\media_data_global_mapped.csv"
# save to CSV
data.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ File written to:\n{output_path}")
