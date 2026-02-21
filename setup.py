import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
    
    
__version__ = "0.0.0"

REPO_NAME = "renal_tumour_project"
AUTHOR_USER_NAME = "MicroPyscho"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "kelechi.ocollins@gmail.com"


setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "An end to end ML Project for predicting kidney (or renal) tumour",
    long_description = long_description,
    long_description_content = "text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {
        "Bug tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")        
)