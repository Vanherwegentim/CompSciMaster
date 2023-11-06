

# MealUpdateRequest

Request body for a meal
## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**name** | **String** | The name of the meal | 
**price** | **Double** | The price of the meal |  [optional]
**kcal** | **Integer** | The energetic value of the meal |  [optional]
**description** | **String** | A description of the meal |  [optional]
**mealType** | [**MealTypeEnum**](#MealTypeEnum) | The type of meal | 



## Enum: MealTypeEnum

Name | Value
---- | -----
VEGAN | &quot;VEGAN&quot;
VEGGIE | &quot;VEGGIE&quot;
MEAT | &quot;MEAT&quot;
FISH | &quot;FISH&quot;



