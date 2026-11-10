"""고정 설정 (희성 결과 JSON의 settings 값과 동일하게 맞춤)."""

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

JUDGE_REPO = "Qwen/Qwen3-8B"
JUDGE_REV = "b968826d9c46dd6066d109eabc6255188de91218"

DS_REPO = "MMMU/MMMU"
DS_REV = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"

CONFIGS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]

SEED = 3407
MIN_PIXELS = 1003520
MAX_PIXELS = 4014080
MAX_IMAGES = 5
MAX_NEW_TOKENS = 8192 #originally 32768
MAX_MODEL_LEN = 64000
