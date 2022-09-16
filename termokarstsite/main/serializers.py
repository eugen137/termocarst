from rest_framework import serializers

from .models import TestPlot, DataValues


class ValueSerializerWrite(serializers.ModelSerializer):
    def create(self, validated_data):
        v = DataValues(**validated_data)
        v.save()
        return DataValues(**validated_data)

    class Meta:
        model = DataValues
        fields = ('date', 'value', 'typedatacol', 'type', 'source', 'testplot', 'ismain')


class DataValuesSerializer(serializers.ModelSerializer):
    typedatacol = serializers.SlugRelatedField(
        read_only=True,
        slug_field='name'
    )
    type = serializers.SlugRelatedField(
        read_only=True,
        slug_field='name'
    )

    class Meta:
        model = DataValues
        fields = ('type', 'typedatacol', 'date', 'value')


class TestPlotListSerilizer(serializers.ModelSerializer):
    zonetype = serializers.SlugRelatedField(
        read_only=True,
        slug_field='name'
    )

    zone = serializers.SlugRelatedField(
        read_only=True,
        slug_field='name'
    )

    class Meta:
        model = TestPlot
        fields = ('id', 'name', 'latitude', 'longitude', 'zone', 'zonetype')


class TestPlotListSerilizerWrite(serializers.ModelSerializer):
    def create(self, validated_data):
        t = TestPlot(**validated_data)
        t.save()
        return TestPlot(**validated_data)

    class Meta:
        model = TestPlot
        fields = ('id', 'name', 'latitude', 'longitude', 'zone', 'zonetype')


class DataSerialize(serializers.Serializer):
    year = serializers.IntegerField()
    square_value = serializers.FloatField()
    square_source = serializers.CharField()

    temp_value = serializers.FloatField()
    temp_source = serializers.CharField()
    perc_value = serializers.FloatField()
    perc_source = serializers.CharField()


class TestPlotDataSerilizer(serializers.Serializer):
    data = DataSerialize(many=True)
