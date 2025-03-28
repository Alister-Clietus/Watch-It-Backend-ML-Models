import pkg_resources

# List of required packages and versions
required_packages = {
    "absl-py": "2.1.0",
    "aiohappyeyeballs": "2.4.6",
    "aiohttp": "3.11.12",
    "aiosignal": "1.3.2",
    "asttokens": "3.0.0",
    "astunparse": "1.6.3",
    "async-timeout": "5.0.1",
    "attrs": "25.1.0",
    "blinker": "1.9.0",
    "certifi": "2025.1.31",
    "charset-normalizer": "3.4.1",
    "click": "8.1.8",
    "colorama": "0.4.6",
    "contourpy": "1.3.1",
    "cycler": "0.12.1",
    "decorator": "5.1.1",
    "exceptiongroup": "1.2.2",
    "executing": "2.2.0",
    "filelock": "3.17.0",
    "Flask": "3.1.0",
    "flatbuffers": "25.2.10",
    "fonttools": "4.56.0",
    "frozenlist": "1.5.0",
    "fsspec": "2025.2.0",
    "gast": "0.6.0",
    "google-pasta": "0.2.0",
    "greenlet": "3.1.1",
    "grpcio": "1.70.0",
    "h5py": "3.12.1",
    "idna": "3.10",
    "intel-openmp": "2021.4.0",
    "ipython": "8.32.0",
    "itsdangerous": "2.2.0",
    "jedi": "0.19.2",
    "Jinja2": "3.1.5",
    "joblib": "1.4.2",
    "keras": "3.8.0",
    "kiwisolver": "1.4.8",
    "libclang": "18.1.1",
    "Markdown": "3.7",
    "markdown-it-py": "3.0.0",
    "MarkupSafe": "3.0.2",
    "matplotlib": "3.10.1",
    "matplotlib-inline": "0.1.7",
    "mdurl": "0.1.2",
    "mkl": "2021.4.0",
    "ml-dtypes": "0.4.1",
    "mpmath": "1.3.0",
    "multidict": "6.1.0",
    "mysql-connector-python": "9.2.0",
    "namex": "0.0.8",
    "networkx": "3.4.2",
    "numpy": "1.26.4",
    "opencv-python": "4.11.0.86",
    "opt_einsum": "3.4.0",
    "optree": "0.14.0",
    "packaging": "24.2",
    "pandas": "2.2.3",
    "parso": "0.8.4",
    "pillow": "11.1.0",
    "prompt_toolkit": "3.0.50",
    "propcache": "0.2.1",
    "protobuf": "5.29.3",
    "psutil": "7.0.0",
    "pure_eval": "0.2.3",
    "Pygments": "2.19.1",
    "pyparsing": "3.2.1",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2025.1",
    "PyYAML": "6.0.2",
    "requests": "2.32.3",
    "rich": "13.9.4",
    "scikit-learn": "1.6.1",
    "scipy": "1.15.2",
    "seaborn": "0.13.2",
    "six": "1.17.0",
    "SQLAlchemy": "2.0.38",
    "stack-data": "0.6.3",
    "sympy": "1.13.1",
    "tbb": "2021.13.1",
    "telepot": "12.7",
    "tensorboard": "2.18.0",
    "tensorboard-data-server": "0.7.2",
    "tensorflow": "2.18.0",
    "tensorflow-io-gcs-filesystem": "0.31.0",
    "tensorflow_intel": "2.18.0",
    "termcolor": "2.5.0",
    "thop": "0.1.1.post2209072238",
    "threadpoolctl": "3.5.0",
    "torch": "2.6.0+cu118",
    "torchaudio": "2.6.0+cu118",
    "torchmetrics": "0.10.3",
    "torchvision": "0.21.0+cu118",
    "tqdm": "4.67.1",
    "traitlets": "5.14.3",
    "typing_extensions": "4.12.2",
    "tzdata": "2025.1",
    "urllib3": "2.3.0",
    "wcwidth": "0.2.13",
    "Werkzeug": "3.1.3",
    "wrapt": "1.17.2",
    "yarl": "1.18.3",
    "opencv-python-headless": "4.11.0.86"
}

def check_installed_packages():
    missing_packages = []
    incorrect_versions = []

    for package, required_version in required_packages.items():
        try:
            installed_version = pkg_resources.get_distribution(package).version
            if installed_version != required_version:
                incorrect_versions.append((package, installed_version, required_version))
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)

    # Display results
    if missing_packages:
        print("\n🚨 Missing Packages:")
        for pkg in missing_packages:
            print(f" - {pkg} (Required: {required_packages[pkg]})")

    if incorrect_versions:
        print("\n⚠️ Incorrect Versions:")
        for pkg, installed, required in incorrect_versions:
            print(f" - {pkg}: Installed {installed}, Required {required}")

    if not missing_packages and not incorrect_versions:
        print("\n✅ All packages are correctly installed!")

if __name__ == "__main__":
    check_installed_packages()
