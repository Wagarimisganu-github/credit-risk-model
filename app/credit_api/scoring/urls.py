from django.urls import path
from .views import predict_credit_risk, home

urlpatterns = [
    path('api/predict/', predict_credit_risk, name='predict_credit_risk'),
    path('', home, name='home'),
]
