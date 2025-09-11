import pandas as pd

# Read in the datasets
global_data = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_250605\Genome-pipeline-code\Genome-pipeline-code\global_platform_data\Global_data.csv")
global_insurance_data = global_data.loc[global_data["Sector"] == "Insurance"]
insurance_bespoke = pd.read_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\global_platform_data\insurance_data.csv")
# 1. Find extra columns
extra_cols = set(insurance_bespoke.columns) - set(global_insurance_data.columns)
print("Extra columns:", extra_cols)
# 2. Common columns, preserving the order from global_insurance_data
common_cols = [col for col in global_insurance_data.columns if col in insurance_bespoke.columns]

# 3. Merge (stack) vertically
merged = pd.concat(
    [global_insurance_data[common_cols], insurance_bespoke[common_cols]],
    axis=0
)

print(merged.shape)
print(merged.head())

# === Add insurer types (P&C, Life, Multiline) ===
import re
import unicodedata

# 1) Normalizer to make matching robust across spellings/encodings/suffixes
_STOPWORDS = {
    'limited','ltd','plc','inc','incorporated','company','companies','co','corp','corporation',
    'group','holdings','holding','public','sa','nv','ag','se','bv','ab','spa','s','p','a','oyj',
    'cooperative','assicurazioni','aktiengesellschaft','gesellschaft','in','of','for','the','and'
}

def _fix_mojibake(s: str) -> str:
    """Fix a few very common mojibake sequences seen in exported CSVs."""
    if not isinstance(s, str):
        return s
    repl = {
        'Ã¼':'ü','Ãœ':'Ü','Ã¶':'ö','Ã–':'Ö','Ã¤':'ä','Ã„':'Ä','ÃŸ':'ß',
        'Ã¡':'á','Ã©':'é','Ãñ':'ñ','Ã±':'ñ','Ã³':'ó','Ã¨':'è','Ãº':'ú'
    }
    for k,v in repl.items():
        s = s.replace(k, v)
    return s

def normalize_company_name(name: str) -> str:
    if not isinstance(name, str):
        return ''
    s = _fix_mojibake(name)
    s = s.replace('&', ' and ')
    # strip accents (e.g., Rück -> Ruck; Münchener -> Munchener)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # keep alphanumerics as tokens
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    tokens = [t for t in s.split() if t not in _STOPWORDS]
    return ' '.join(tokens).strip()

# 2) Exact map for cases where a single token uniquely identifies the brand
#    Keys are *normalized* names (use `normalize_company_name` on the left-hand side).
_EXACT_MAP = {
    # Australia
    normalize_company_name('AUB Group'): 'Multiline',
    normalize_company_name('Challenger'): 'Life',
    normalize_company_name('Helia Group'): 'P&C',
    normalize_company_name('Insurance Australia Group'): 'P&C',
    normalize_company_name('Medibank'): 'Life',
    normalize_company_name('nib holdings'): 'Life',
    normalize_company_name('QBE Insurance Group'): 'P&C',
    normalize_company_name('Steadfast Group'): 'Multiline',
    normalize_company_name('Suncorp Group'): 'P&C',

    # China / HK
    normalize_company_name('Ping An Insurance'): 'Multiline',
    normalize_company_name("The People's Insurance Company of China"): 'Multiline',
    normalize_company_name('New China Life Insurance'): 'Life',
    normalize_company_name('China Pacific Insurance'): 'Multiline',
    normalize_company_name('China Life Insurance'): 'Life',
    normalize_company_name('PICC Property & Casualty'): 'P&C',
    normalize_company_name('ZhongAn Online P&C Insurance'): 'P&C',
    normalize_company_name('China Taiping Insurance (Life)'): 'Life',

    # Japan & Korea
    normalize_company_name('MS&AD Insurance Group'): 'Multiline',
    normalize_company_name('Sompo Holdings'): 'Multiline',
    normalize_company_name('Tokio Marine Holdings'): 'Multiline',
    normalize_company_name('Dai-ichi Life Holdings'): 'Life',
    normalize_company_name('T&D Holdings'): 'Life',
    normalize_company_name('Japan Post Insurance'): 'Life',
    normalize_company_name('Samsung Fire & Marine Insurance'): 'P&C',
    normalize_company_name('Hyundai Marine & Fire Insurance'): 'P&C',
    normalize_company_name('DB Insurance'): 'P&C',
    normalize_company_name('Samsung Life Insurance'): 'Life',
    normalize_company_name('Hanwha Life Insurance'): 'Life',
    normalize_company_name('Hanwha General Insurance'): 'P&C',

    # Europe (pan‑EU/UK/CH/NL/IT/DE/DK/BE/PL/AT/ES)
    normalize_company_name('Ageas'): 'Multiline',
    normalize_company_name('AXA'): 'Multiline',
    normalize_company_name('Allianz'): 'Multiline',
    normalize_company_name('Assicurazioni Generali'): 'Multiline',
    normalize_company_name('Generali Group'): 'Multiline',
    normalize_company_name('Zurich Insurance Group'): 'Multiline',
    normalize_company_name('Swiss Life Holding'): 'Life',
    normalize_company_name('Swiss Re'): 'Multiline',
    normalize_company_name('Munich Re'): 'Multiline',
    normalize_company_name('Muenchener Rueckversicherungs Gesellschaft'): 'Multiline',
    normalize_company_name('Hannover Rueck'): 'Multiline',
    normalize_company_name('Aegon'): 'Life',
    normalize_company_name('ASR Nederland'): 'Multiline',
    normalize_company_name('NN Group'): 'Multiline',
    normalize_company_name('Tryg'): 'P&C',
    normalize_company_name('Aviva'): 'Multiline',
    normalize_company_name('Admiral Group'): 'P&C',
    normalize_company_name('Beazley'): 'P&C',
    normalize_company_name('Hiscox'): 'P&C',
    normalize_company_name('Legal & General Group'): 'Life',
    normalize_company_name('M&G'): 'Life',
    normalize_company_name('Phoenix Group Holdings'): 'Life',
    normalize_company_name('Talanx AG'): 'Multiline',
    normalize_company_name('Helvetia Holding'): 'Multiline',
    normalize_company_name('Baloise Holding'): 'Multiline',
    normalize_company_name('Mapfre'): 'Multiline',
    normalize_company_name('PZU SA'): 'P&C',
    normalize_company_name('Vienna Insurance Group'): 'Multiline',
    normalize_company_name('ASR Nederland N.V.'): 'Multiline',
    normalize_company_name('Ageas SA/NV'): 'Multiline',
    normalize_company_name('Tryg A/S'): 'P&C',

    # Italy
    normalize_company_name('Poste Italiane'): 'Multiline',
    normalize_company_name('Unipol Assicurazioni'): 'Multiline',

    # Saudi Arabia & GCC
    normalize_company_name('The Company for Cooperative Insurance'): 'Multiline',  # Tawuniya
    normalize_company_name('Malath Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Mediterranean and Gulf Cooperative Insurance and Reinsurance Company'): 'Multiline', # MedGulf
    normalize_company_name('Mutakamela Insurance Company'): 'Multiline',
    normalize_company_name('Salama Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Walaa Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Arabian Shield Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Saudi Arabian Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Gulf Union Alahlia Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Arabia Insurance Cooperative Company'): 'Multiline',
    normalize_company_name('Al Etihad Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Al Sagr Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('United Cooperative Assurance Company'): 'Multiline',
    normalize_company_name('Saudi Reinsurance Company'): 'Multiline',
    normalize_company_name('Bupa Arabia for Cooperative Insurance Company'): 'Life',
    normalize_company_name('Al Rajhi Company for Cooperative Insurance'): 'Multiline',
    normalize_company_name('Chubb Arabia Cooperative Insurance Company'): 'P&C',
    normalize_company_name('Gulf Insurance Group'): 'Multiline',
    normalize_company_name('Gulf General Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Buruj Cooperative Insurance Company'): 'Multiline',
    normalize_company_name('Liva Insurance Company'): 'Multiline',
    normalize_company_name('Wataniya Insurance Company'): 'Multiline',
    normalize_company_name('Saudi Enaya Cooperative Insurance Company'): 'Life',
    normalize_company_name('Aljazira Takaful Taawuni Company'): 'Life',
    normalize_company_name('Allied Cooperative Insurance Group'): 'Multiline',
    normalize_company_name('Amana Cooperative Insurance Company'): 'Multiline',

    # SE Asia
    normalize_company_name('Thai Life Insurance'): 'Life',
    normalize_company_name('Bangkok Life Assurance'): 'Life',
    normalize_company_name('Bangkok Insurance PCL'): 'Multiline',  # composite
    normalize_company_name('Great Eastern Holdings'): 'Life',
    normalize_company_name('Bao Viet Holdings'): 'Multiline',
    normalize_company_name('Dhipaya Group Holdings'): 'Multiline',  # group incl. life/general
    normalize_company_name('Syarikat Takaful Malaysia'): 'Multiline',
    normalize_company_name('PT Asuransi Tugu Pratama Indonesia'): 'P&C',

    # North America
    normalize_company_name('Aflac'): 'Life',
    normalize_company_name('Assurant'): 'P&C',
    normalize_company_name('Arthur J. Gallagher'): 'Multiline',
    normalize_company_name('Aon'): 'Multiline',
    normalize_company_name('Brown & Brown'): 'Multiline',
    normalize_company_name('Chubb'): 'P&C',
    normalize_company_name('Cincinnati Financial'): 'P&C',
    normalize_company_name('Erie Indemnity'): 'P&C',
    normalize_company_name('Everest Group'): 'P&C',
    normalize_company_name('Globe Life'): 'Life',
    normalize_company_name('Hartford Financial Services'): 'Multiline',
    normalize_company_name('The Hartford Insurance Group'): 'Multiline',
    normalize_company_name('Loews'): 'P&C',
    normalize_company_name('Marsh & McLennan'): 'Multiline',
    normalize_company_name('MetLife'): 'Life',
    normalize_company_name('Principal Financial Group'): 'Life',
    normalize_company_name('Prudential Financial'): 'Life',
    normalize_company_name('The Allstate Corporation'): 'P&C',
    normalize_company_name('Allstate Corporation'): 'P&C',
    normalize_company_name('Progressive Corporation'): 'P&C',
    normalize_company_name('The Travelers Companies'): 'P&C',
    normalize_company_name('Travelers'): 'P&C',
    normalize_company_name('CNA Financial'): 'P&C',
    normalize_company_name('W. R. Berkley'): 'P&C',
    normalize_company_name('American International Group'): 'Multiline',
    normalize_company_name('Unum Group'): 'Life',
    normalize_company_name('Lincoln National Corporation'): 'Life',
    normalize_company_name('Willis Towers Watson'): 'Multiline',

    # Global life groups (duplicates/variants)
    normalize_company_name('AIA Group'): 'Life',
    normalize_company_name('Manulife Financial'): 'Life',
    normalize_company_name('Sun Life Financial'): 'Life',
    normalize_company_name('Phoenix Group Holdings plc'): 'Life',
    normalize_company_name('Legal & General Group Plc'): 'Life',
    normalize_company_name('Prudential plc'): 'Life',
    normalize_company_name('M&G plc'): 'Life',

    # India / Taiwan
    normalize_company_name('HDFC Life Insurance'): 'Life',
    normalize_company_name('ICICI Prudential Life'): 'Life',
    normalize_company_name('SBI Life Insurance'): 'Life',
    normalize_company_name('LIC'): 'Life',
    normalize_company_name('Huaxia'): 'Life',         # Huaxia Life
    normalize_company_name('Cathay Life'): 'Life',
    normalize_company_name('Fubon Financial'): 'Multiline',

    # Health / managed care (treated as Life/Health)
    normalize_company_name('Cigna'): 'Life',
    normalize_company_name('United Health'): 'Life',
    normalize_company_name('UnitedHealth'): 'Life',
    normalize_company_name('Humana'): 'Life',
    normalize_company_name('CVS Health'): 'Life',
}

# 3) Pattern map for broader catch‑alls and spelling variants
_PATTERN_MAP = [
    (re.compile(r'\bm[ue]nchener\s+r[ue]ck', re.I), 'Multiline'),   # Münchener Rück
    (re.compile(r'\bhannover\s+r[ue]ck', re.I), 'Multiline'),
    (re.compile(r'\bswiss\s+re\b', re.I), 'Multiline'),
    (re.compile(r'\bproperty\b.*\bcasualty\b', re.I), 'P&C'),
    (re.compile(r'\bp\s*(&|and)?\s*c\b', re.I), 'P&C'),
    (re.compile(r'\bfire\b|\bmarine\b|\bgeneral\s+insurance\b', re.I), 'P&C'),
    (re.compile(r'\blife\b|\bassurance\b|\bhealth\b', re.I), 'Life'),
    # Brokers/intermediaries -> Multiline
    (re.compile(r'\baon\b|\bmarsh\b|\bmclennan\b|\bwillis\s+towers\b|\bwtw\b|\barthur\s+j\b|\bbrown\s+and\s+brown\b', re.I), 'Multiline'),
    # Japan big three (redundant safety)
    (re.compile(r'\b(ms\s*&?\s*ad|mitsui\s+sumitomo|sompo|tokio\s+marine)\b', re.I), 'Multiline'),
    # GCC "cooperative insurance" (default composite)
    (re.compile(r'\bcooperative\s+insurance\b|\btakaful\b', re.I), 'Multiline'),
    # Health specialists that should stay Life even if "cooperative"
    (re.compile(r'\bbupa\s+arabia\b|\bsaudi\s+enaya\b|\baljazira\s+takaful\b', re.I), 'Life'),
]

def classify_insurer(raw_name: str) -> str:
    n = normalize_company_name(raw_name)

    # 1) Exact map
    if n in _EXACT_MAP:
        return _EXACT_MAP[n]

    # 2) Pattern rules (ordered)
    for pat, label in _PATTERN_MAP:
        if pat.search(raw_name) or pat.search(n):
            return label

    # 3) Fallbacks for a few frequent brand stems
    stems = {
        'aia':'Life', 'aegon':'Life', 'ageas':'Multiline', 'axa':'Multiline', 'allianz':'Multiline',
        'generali':'Multiline', 'zurich':'Multiline', 'tryg':'P&C', 'aviva':'Multiline',
        'admiral':'P&C', 'beazley':'P&C', 'hiscox':'P&C', 'phoenix':'Life', 'legal general':'Life',
        'mapfre':'Multiline', 'helvetia':'Multiline', 'baloise':'Multiline', 'qbe':'P&C',
        'iag':'P&C', 'suncorp':'P&C', 'pzu':'P&C', 'vienna insurance':'Multiline',
        'manulife':'Life', 'sun life':'Life', 'great eastern':'Life',
        'prudential financial':'Life', 'prudential plc':'Life', 'metlife':'Life',
        'principal financial':'Life', 'unum':'Life', 'lincoln national':'Life',
        'travelers':'P&C', 'progressive':'P&C', 'allstate':'P&C', 'cna financial':'P&C',
        'berkley':'P&C', 'cincinnati financial':'P&C', 'erie indemnity':'P&C',
        'arch capital':'P&C', 'everest':'P&C', 'loews':'P&C'
    }
    for k,v in stems.items():
        if k in n:
            return v

    return 'Unknown'

# 4) Apply to your merged dataframe
merged['Type_of_Insurer'] = merged['Company_name'].apply(classify_insurer)

# 5) Quick QA: list any that didn’t map (so you can add to _EXACT_MAP as needed)
unknowns = (merged.loc[merged['Type_of_Insurer'].eq('Unknown'), 'Company_name']
                 .dropna().drop_duplicates().sort_values())
print(f"Unknown / needs review ({len(unknowns)}):")
for u in unknowns:
    print("  -", u)

# 6) (Optional) sanity check counts
print(merged['Type_of_Insurer'].value_counts(dropna=False))

# 7) Save
# merged.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome.csv", index=False)
merged.to_csv(r"C:\Users\60848\OneDrive - Bain\Desktop\Project_Genome\casework\AIA\Clean\Clean_insurance_data_genome_gpt.csv")
