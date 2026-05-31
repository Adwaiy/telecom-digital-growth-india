import streamlit as st
import pandas as pd

st.title("Telecom Dashboard Test")

df = pd.read_csv("telecom_project_data.csv")

st.write(df.head())

st.write(df.columns)
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Average Literacy",
        round(df['literacy'].mean(), 2)
    )

with col2:
    st.metric(
        "Average Teledensity",
        round(df['TELEDENSITY'].mean(), 2)
    )
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.scatter(
    df['TELEDENSITY'],
    df['e_transactions_per_1000']
)

ax.set_xlabel("Teledensity")
ax.set_ylabel("Digital Transactions")

st.pyplot(fig)
fig2, ax2 = plt.subplots(figsize=(10,5))

sorted_df = df.sort_values(
    by='e_transactions_per_1000',
    ascending=False
)

ax2.bar(
    sorted_df['state'],
    sorted_df['e_transactions_per_1000']
)

plt.xticks(rotation=90)

st.pyplot(fig2)
import seaborn as sns

numeric_cols = [
    'literacy',
    'TELEDENSITY',
    'subscriber_growth_rate',
    'e_transactions_per_1000'
]

corr = df[numeric_cols].corr()

fig3, ax3 = plt.subplots(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    ax=ax3
)

st.pyplot(fig3)
st.subheader("Model Performance Summary")

results_df = pd.DataFrame({
    'Model': [
        'OLS',
        'Elastic Net',
        'Random Forest',
        'XGBoost'
    ],
    'Key Finding': [
        'Literacy strongest predictor',
        'Sparse telecom coefficients',
        'Nonlinear relationships detected',
        'Overfitting under small sample'
    ]
})

st.dataframe(results_df)
st.subheader("State Comparison")

state1 = st.selectbox(
    "Select State 1",
    df['state'].unique()
)

state2 = st.selectbox(
    "Select State 2",
    df['state'].unique(),
    index=1
)

comparison = df[
    df['state'].isin([state1, state2])
]

st.dataframe(comparison)
top_states = df.nlargest(
    5,
    'e_transactions_per_1000'
)[['state', 'e_transactions_per_1000']]

st.subheader("Top 5 Digital Economy States")

st.table(top_states)
x_feature = st.selectbox(
    "Choose Feature",
    [
        'literacy',
        'TELEDENSITY',
        'subscriber_growth_rate',
        'DCI_PCA'
    ]
)
st.markdown("""
### Cluster Interpretation

- Cluster 0 → Digitally advanced states
- Cluster 1 → Emerging telecom states
- Cluster 2 → Underserved states

This highlights structural heterogeneity across India.
""")
csv = df.to_csv(index=False)

st.download_button(
    label="Download Dataset",
    data=csv,
    file_name='telecom_data.csv',
    mime='text/csv'
)
st.subheader("Project Workflow")

st.markdown("""
1. Data Collection  
2. Data Cleaning  
3. Feature Engineering  
4. PCA-based DCI Construction  
5. Econometric Modeling  
6. Machine Learning Analysis  
7. State Clustering  
8. Dashboard Deployment
""")
st.subheader("Research Limitations")

st.markdown("""
- Small sample size (~36 states)
- Cross-sectional rather than panel data
- Telecom variables exhibit multicollinearity
- Machine learning results treated as exploratory
- No causal inference framework
""")
st.info(
    "Key Insight: Literacy consistently outweighed telecom infrastructure in explaining digital economic participation."
)
