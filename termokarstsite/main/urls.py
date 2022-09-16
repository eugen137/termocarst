from django.urls import path, include
from . import views


app_name = 'main'

urlpatterns = [
    path('', views.tu_url, name='index'),
    path('tu/', views.tu_url, name='tu_url'),
    path('get_tu_for_zone/', views.get_tu_for_zone, name='get_tu_for_zone'),
    # path('get_square/', views.SquareList.as_view(), name='get_square_url'),
    path('api_test_plot/', views.ApiTestPlotList.as_view(), name='api_test_plot_url'),
    path('get_test_plot_data/', views.TestPlotListData.as_view(), name='get_test_plot_url'),
    path('post_value_data/', views.ValuesFromExcel.as_view(), name='post_value_data_url'),
    path('test_plot_from_file/', views.TestPlotsFromExcel.as_view(), name='test_plot_from_file_url'),
]
