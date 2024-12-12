# _The New Satire Times_: A Vertical Search Engine for Satirical News Articles

- **Setup**: Create a virtual environment (Python 3.11.2):
  ```
  > <path of your python installation>/python -m venv si650
  > si650\Scripts\activate (Windows) or si650/bin/activate (Mac & others)
  > python -m pip install -r requirements.txt
  ```
- Download the data files from [this folder](https://drive.google.com/drive/folders/1JHmjQhL49yYWYSZqT6rPh2q3xWDy2O0N) and place them under `data/`:
    - `processed_articles_dedup_nsfwtags_sarcasm.csv`
    - `stopwords_updated.txt`
    - `multiword_expressions.csv`
    - `relevance_train.csv`
    - `relevance_test.csv`
    - `relevance_dev.csv`
    - `body_embeddings.npy`
- Navigate to `app/` and run this command to launch the search engine in the browser (http://127.0.0.1:8000/):
  ```
  > python -m uvicorn app:app
  ```
- The default ranker is BM25.
- Check the 'Safe Search' box to filter out explicit results.


<img src="https://github.com/user-attachments/assets/88d49d68-951a-49bb-99b1-fec73bef5b9b" alt="drawing" width="500"/>




