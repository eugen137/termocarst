from .models import TestPlot, Zone
from django import forms


class SelectTestPlot(forms.Form):
    def __init__(self, *args, **kwargs):
        self.choices_test_plots = kwargs.pop('choices_test_plots')
        super(SelectTestPlot, self).__init__(*args, **kwargs)
        self.fields['test_plot'].choices = self.choices_test_plots
    zone = forms.ChoiceField(label="Выберите зону", choices=Zone.objects.all().values_list("name", "name"))
    test_plot = forms.ChoiceField(label="Выберите тестовый участок")