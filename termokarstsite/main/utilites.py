from django.db.models import Q

from main.models import DataValues, TypeParameters, TestPlot, Zone
from main.serializers import TestPlotDataSerilizer, DataSerialize


def get_test_plot_data(test_plot, zone):
    test_plot = TestPlot.objects.filter(zone__name__icontains=zone).filter(name=test_plot)[0]
    square = DataValues.objects.filter(testplot=test_plot, type__name="Площадь")
    perc = DataValues.objects.filter(testplot=test_plot, type__name="Осадки")
    temp = DataValues.objects.filter(testplot=test_plot, type__name="Температура")

    dates_square = list(square.values_list('date'))
    dates_square = list(set(dates_square))

    dates_perc = list(perc.values_list('date'))
    dates_perc = list(set(dates_perc))

    dates_temp = list(temp.values_list('date'))
    dates_temp = list(set(dates_temp))

    # удаление дубликатов
    dates = []
    dates.extend(dates_square)
    dates.extend(dates_perc)
    dates.extend(dates_temp)

    dates.sort()

    dates = list(dict.fromkeys(dates))
    data_ser = []

    for d in dates:
        val = []
        param = [square, temp, perc]
        for par in param:
            tmp = par.filter(date=d[0])
            tmp_val = tmp_sour = None
            if tmp:
                tmp_val = tmp.values_list('value')[0][0]
                tmp_sour = tmp.values_list('source__name')[0][0]
            val.append({'val': tmp_val, 'source': tmp_sour})
        data_ser.append({'year': d[0].year,
                         'square_value': val[0]['val'], 'square_source': val[0]['source'],
                         'temp_value': val[1]['val'], 'temp_source': val[1]['source'],
                         'perc_value': val[2]['val'], 'perc_source': val[2]['source']})
    serializer_tu1 = DataSerialize(data=data_ser, many=True)
    serializer_tu1.is_valid()
    return serializer_tu1

