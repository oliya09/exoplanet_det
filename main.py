from pipeline.run_pipeline import run_pipeline

if __name__ == "__main__":
    tic_ids = [
"TIC 458392017",
"TIC 120384930",
"TIC 276593842",
"TIC 392047159",
"TIC 540038274",
"TIC 135902118",
"TIC 308472091",
"TIC 281940623",
"TIC 439581206",
"TIC 199340578",
"TIC 373924059",
"TIC 219875346",
"TIC 119284507",
"TIC 375021884",
"TIC 202498061",
"TIC 341082739",
"TIC 495733120",
"TIC 237913194",  
"TIC 172900988",  
"TIC 4672985",    
"TIC 46432937",   
  
]
    run_pipeline(tic_ids)
    # For demo: streamlit run demo.py