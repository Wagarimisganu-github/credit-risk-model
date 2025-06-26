from django import forms

class PredictionForm(forms.Form):
    # Numerical fields
    TransactionMonth = forms.FloatField(label="Transaction Month", required=True)
    TotalTransactionAmount = forms.FloatField(label="Total Transaction Amount", required=True)
    TransactionCount = forms.FloatField(label="Transaction Count", required=True)
    AverageTransactionAmount = forms.FloatField(label="Average Transaction Amount", required=True)
    TransactionHour = forms.FloatField(label="Transaction Hour", required=True)
    Amount = forms.FloatField(label="Amount", required=True)
    TransactionYear = forms.FloatField(label="Transaction Year", required=True)
    TransactionDay = forms.FloatField(label="Transaction Day", required=True)
    Value = forms.FloatField(label="Value", required=True)
    StdDevTransactionAmount = forms.FloatField(label="Standard Deviation of Transaction Amount", required=True)
    MonetaryTotal_woe = forms.FloatField(label="Monetary Total WoE", required=True)
    Frequency_woe = forms.FloatField(label="Frequency WoE", required=True)
    MonetaryAvg_woe = forms.FloatField(label="Monetary Average WoE", required=True)
    Recency_woe = forms.FloatField(label="Recency WoE", required=True)
    
    # Categorical fields (using choices for ProductCategory, ChannelId, PricingStrategy)
    ProductCategory = forms.ChoiceField(choices=[(1, 'financial_services'), (2, 'airtime'),(3,'utility_bill'),(4,'data_bundles'),(5,'tv'),(6,'ticket'),(7,'movies'),(8,'transport'),(8,'other')], label="Product Category", required=True)
    ChannelId = forms.ChoiceField(choices=[(1, 'ChannelId_1'), (2, 'ChannelId_2'), (3, 'ChannelId_3'),(4, 'ChannelId_5')], label="Channel ID", required=True)
    PricingStrategy = forms.ChoiceField(choices=[(1, 'Strategy 1'), (2, 'Strategy 2'),(3, 'Strategy 3'),(4, 'Strategy 4')], label="Pricing Strategy", required=True)
    FraudResult = forms.ChoiceField(choices=[(0, 'No Fraud'), (1, 'Fraud')], label="Fraud Result", required=True)
