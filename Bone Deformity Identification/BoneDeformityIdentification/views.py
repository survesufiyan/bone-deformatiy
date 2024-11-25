from django.shortcuts import render
from users.forms import UserRegistrationForm



def index(request):
    return render(request,'index.html')

def user(request):
    return render(request,'UserLogin.html')

def admin(request):
    return render(request,'AdminLogin.html')

def register(request):
 form = UserRegistrationForm()
 return render(request, 'UserRegistrations.html', {'form': form})