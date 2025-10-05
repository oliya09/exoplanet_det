from pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    tic_ids = [
        "TIC 150428135",
        "TIC 219006972",
        "TIC 285853156",
        "TIC 392229331",
        "TIC 393818343",
        "TIC 356473034"
    ]
    run_pipeline(tic_ids)
    # Для демо: streamlit run pipeline/demo.py