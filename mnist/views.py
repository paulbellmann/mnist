# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render, HttpResponse, redirect
from django.views.decorators.csrf import csrf_exempt

import numpy as np
import pickle

# filename = 'finalized_model.sav'
# clf = pickle.load(open(filename, 'rb'))

# Create your views here.
def index(request):
    context = {
        'title': 'Hello'
    }
    return render(request, 'index.html', context)

@csrf_exempt
def predict(request):
    if request.method == "POST":
        a = request.POST['arr']
        a = eval(a)
        example_digit = np.asarray(a)
        example_digit = example_digit.reshape(1, -1)

        prediction = clf.predict(example_digit)[0]
        return HttpResponse(prediction)