from django.http import request
from django.shortcuts import render
import joblib
def index(request):
    text=""
    msg=""
    model=joblib.load('final_model.sav')
    if request.method=='POST':
        msg=request.POST['msg']
        print(msg)
        text=model.predict([msg])
    return render(request,'index.html',{'text':text,'msg':msg})
