## Setup
Fork copy to local 

```
pip install -r requirements.txt
```

## Run
```
streamlit run main.py
```

## Usage
1. Select ticker
2. Select start and end time
3. Select rolling window
4. OPTIONAL: Calculate optimal number of clusters
5. Apply clustering model on prices and alternative indicators (VIX and Volume by default. No support for other indicators yet)
![Alt text](https://github.com/AnilabsWebTeam/st-fintech/blob/main/assets/imgs/1.png)
6. Select forward and backward returns by period, labeled by regime
![Alt text](https://github.com/AnilabsWebTeam/st-fintech/blob/main/assets/imgs/2.png)
7. ??? 
8. Profit