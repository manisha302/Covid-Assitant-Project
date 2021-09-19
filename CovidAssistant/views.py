from django.shortcuts import render
import pickle

# scaler = pickle.load(open("Scaler.pkl", "rb"))
model_KNN = pickle.load(open('CovidAssistant/knn_model.pkl', 'rb'))


def home(request):
    return render(request, "home.html")


# our result page view
def predict(request):

    gdp_per_capita = int(request.POST['gdp_per_capita'])
    print(gdp_per_capita)
    a = int(request.POST['stringency_index'])
    b = int(request.POST['reproduction_rate'])
    c = int(request.POST['new_tests_smoothed_per_thousand'])
    d = int(request.POST['population_density'])
    e = int(request.POST['population'])
    f = int(request.POST['new_tests_per_thousand'])
    g = int(request.POST['new_tests_smoothed'])
    h = int(request.POST['total_tests_per_thousand'])
    i = int(request.POST['new_tests'])
    j = int(request.POST['tests_per_case'])
    k = int(request.POST['new_vaccinations'])

    sc = model_KNN.predict([[gdp_per_capita, a, b, c, d, e, f, g, h, i, j, k]])
    sc = str(sc)[1:-1]
    sc = str(sc)[1:-1]

    return render(request, 'result.html', {'Res': sc})
# Create your views here.
