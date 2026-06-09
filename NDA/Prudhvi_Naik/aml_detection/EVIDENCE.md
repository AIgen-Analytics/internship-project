# 🧾 SYSTEM COMPLETION EVIDENCE

## 1. Output CSV Confirmation
- File `data/final_pipeline_outputs.csv` exists: **True**

## 2. First 10 Rows of Output
```text
     Transaction ID  Fraud Risk Score Risk Category       Predicted Typology                                                                                                                 Typology Probability Distribution                                                                                     Key Risk Drivers
TXNEHDI4WKBBG3F31SV              2.50      Low Risk            Charity Abuse                     {'Charity Abuse': np.float64(69.85), 'Structuring (Smurfing)': np.float64(16.5), 'Funnel Account Network': np.float64(11.08)}    transaction_amount (-2.12) | transaction_mode_channel_bank (0.28) | debit_summation_period (0.28)
TXNROIJYWBBLN9AEPNO              1.90      Low Risk Rapid Multi-Hop Layering           {'Rapid Multi-Hop Layering': np.float64(42.02), 'Money Mule Network': np.float64(29.78), 'Circular Transaction Loop': np.float64(6.96)} sender_running_balance_txn_amount (-0.38) | transaction_type_dr_cr (-0.33) | amt_vs_cust_max (-0.28)
TXN4GDF28BMERPF23G3             87.16     High Risk            Charity Abuse                  {'Charity Abuse': np.float64(69.34), 'Funnel Account Network': np.float64(10.31), 'Circular Transaction Loop': np.float64(7.39)}         cp_fraud_rate_hist (0.81) | transaction_amount (0.79) | transaction_mode_channel_bank (0.43)
TXNV0YRZM9A6TG0YJBQ             73.33     High Risk Rapid Multi-Hop Layering             {'Rapid Multi-Hop Layering': np.float64(92.54), 'Circular Transaction Loop': np.float64(6.5), 'Money Mule Network': np.float64(0.48)}         transaction_amount (2.28) | transaction_status (0.29) | transaction_mode_channel_bank (0.26)
TXN0MV6B8P4AKWOA5PN              1.55      Low Risk   Structuring (Smurfing)                       {'Structuring (Smurfing)': np.float64(81.99), 'Charity Abuse': np.float64(9.5), 'Funnel Account Network': np.float64(4.89)}   transaction_amount (-1.60) | transaction_mode_channel_bank (-0.41) | debit_summation_period (0.27)
TXNCVLPLZ8IA4LLA4YM             74.60     High Risk Rapid Multi-Hop Layering {'Rapid Multi-Hop Layering': np.float64(84.74), 'Circular Transaction Loop': np.float64(11.76), 'Underground Banking (Hawala)': np.float64(1.74)}                  transaction_amount (1.91) | acct_fraud_rate_hist (0.35) | transaction_status (0.29)
TXNHRM49180OFMJ7E8W             86.41     High Risk            Charity Abuse                    {'Charity Abuse': np.float64(65.38), 'Third-Party Payment Web': np.float64(17.0), 'Funnel Account Network': np.float64(15.05)}         cp_fraud_rate_hist (0.78) | transaction_mode_channel_bank (0.68) | transaction_amount (0.68)
TXNTWJ03UNU1E9UPR2I              1.25      Low Risk   Funnel Account Network                  {'Funnel Account Network': np.float64(34.39), 'Charity Abuse': np.float64(29.52), 'Pass-Through Transit Hub': np.float64(23.84)}            transaction_amount (-1.92) | acct_fraud_rate_hist (0.54) | debit_summation_period (-0.28)
TXN7859DVHFT89JNL8R              4.37      Low Risk Rapid Multi-Hop Layering            {'Rapid Multi-Hop Layering': np.float64(75.61), 'Money Mule Network': np.float64(13.51), 'Circular Transaction Loop': np.float64(6.9)}       transaction_mode_channel_bank (-0.61) | transaction_amount (0.55) | cp_fraud_rate_hist (-0.30)
TXNG5T2PK76VA3V2JCD              6.01      Low Risk Rapid Multi-Hop Layering              {'Rapid Multi-Hop Layering': np.float64(57.42), 'Funnel Account Network': np.float64(27.37), 'Money Mule Network': np.float64(9.46)}                       transaction_mode_channel_bank (-0.65) | transaction_amount (0.53) | hr (-0.34)
```

## 7. Total Row Count Processed
- **77314** rows processed and scored in the temporal test set (20% of 386k).

## 3. Final Fraud Model Metrics
- **ROC-AUC:** 0.9478
- **PR-AUC:** 0.8468
```text
              precision    recall  f1-score   support

  Legitimate       0.90      0.98      0.94     62890
       Fraud       0.89      0.54      0.67     14424

    accuracy                           0.90     77314
   macro avg       0.89      0.76      0.81     77314
weighted avg       0.90      0.90      0.89     77314

Confusion Matrix:
[[61878  1012]
 [ 6635  7789]]
```

## 4. Final Typology Model Metrics
- **Accuracy:** 0.7807
- **Macro F1:** 0.7295
```text
                              precision    recall  f1-score   support

               Charity Abuse       0.75      0.79      0.77      1368
   Circular Transaction Loop       0.55      0.31      0.40      1036
      Funnel Account Network       0.69      0.51      0.59       840
 High-Risk Corridor Transfer       1.00      1.00      1.00      1625
          Money Mule Network       0.41      0.23      0.29       807
    Pass-Through Transit Hub       0.96      0.91      0.94      1115
    Rapid Multi-Hop Layering       0.72      0.84      0.77      4186
      Structuring (Smurfing)       0.96      0.96      0.96      1481
     Third-Party Payment Web       0.69      0.85      0.76       959
Underground Banking (Hawala)       0.81      0.84      0.82      1007

                    accuracy                           0.78     14424
                   macro avg       0.75      0.72      0.73     14424
                weighted avg       0.77      0.78      0.77     14424

```

## 5. Top 30 Feature Importances
```text
                                 Feature  Importance
                      transaction_amount         838
           transaction_mode_channel_bank         496
                      cp_fraud_rate_hist         316
                    acct_fraud_rate_hist         312
       sender_running_balance_txn_amount         310
                  debit_summation_period         264
                    cp_degree_centrality         244
  sender_cumulative_daily_balance_change         243
                                      hr         238
                receiver_current_balance         231
                  transaction_type_dr_cr         206
                   receiver_country_code         201
                  sender_current_balance         193
                        cp_mean_amt_hist         182
     receiver_bal_ratio_after_to_current         170
           receiver_acct_outflow_amt_30d         155
                      transaction_status         136
                     amt_to_income_ratio         124
       sender_bal_ratio_after_to_current         124
     receiver_running_balance_txn_amount         114
                     amt_deviation_score         113
                             cp_pagerank         110
            receiver_acct_outflow_amt_7d         109
                 credit_summation_period         109
               sender_acct_txn_count_24h         105
                               cash_flag          97
receiver_cumulative_daily_balance_change          96
                     cust_txn_count_hist          95
                     acct_txn_count_hist          87
             sender_acct_inflow_count_7d          84
```

## 6. SHAP Transaction-Level Reconstruction
- **Base Value (Log Odds):** -1.296648
- **Sum of SHAP Contributions:** -2.366172
- **Reconstructed Margin:** -3.662821
- **Reconstructed Probability (Sigmoid):** 0.025018
- **Actual Model Probability:** 0.025018
*(Note: Due to float precision, actual vs reconstructed may differ by ~1e-15, proving exact mathematically additive attribution)*

## 8. Pipeline Runtime
- End-to-End Runtime: **85.93 seconds** (Processing 386570 transactions and training 2 models).
