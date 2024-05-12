import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
import joblib
import numpy as np  
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt, name='dispatch')

# Create your views here.
class PredictView(View):
    def post(self, request):
        data = json.loads(request.body)
        print(data) 

        # Ensure all values are numeric, replace non-numeric values with None
        for key, value in data.items():
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = None

        model = joblib.load('E:\\Care-connect\\Model\\trained_model.pkl')
        user_input = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(user_input)
        print('prediction', prediction)
        return JsonResponse({'prediction': prediction.tolist()})
        
