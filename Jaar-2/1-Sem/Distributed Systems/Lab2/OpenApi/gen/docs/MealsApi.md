# MealsApi

All URIs are relative to *http://localhost:8080*

Method | HTTP request | Description
------------- | ------------- | -------------
[**addMeal**](MealsApi.md#addMeal) | **POST** /meals | Add a new meal
[**deleteMeal**](MealsApi.md#deleteMeal) | **DELETE** /meals/{id} | Remove a meal
[**getMealById**](MealsApi.md#getMealById) | **GET** /meals/{id} | Get a meal by its id
[**getMeals**](MealsApi.md#getMeals) | **GET** /meals | Retrieve all meals
[**updateMeal**](MealsApi.md#updateMeal) | **PUT** /meals/{id} | Update existing meal


<a name="addMeal"></a>
# **addMeal**
> Object addMeal(mealUpdateRequest)

Add a new meal

Add a new meal

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.models.*;
import org.openapitools.client.api.MealsApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost:8080");

    MealsApi apiInstance = new MealsApi(defaultClient);
    MealUpdateRequest mealUpdateRequest = {"name":"Lasagna de la casa","description":"Garfield's favorite dish","kcal":2000,"price":7,"mealType":"MEAT"}; // MealUpdateRequest | 
    try {
      Object result = apiInstance.addMeal(mealUpdateRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling MealsApi#addMeal");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **mealUpdateRequest** | [**MealUpdateRequest**](MealUpdateRequest.md)|  |

### Return type

**Object**

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**201** | New Meal created |  -  |

<a name="deleteMeal"></a>
# **deleteMeal**
> Meal deleteMeal(id)

Remove a meal

Remove an existing meal

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.models.*;
import org.openapitools.client.api.MealsApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost:8080");

    MealsApi apiInstance = new MealsApi(defaultClient);
    UUID id = new UUID(); // UUID | Id of the meal
    try {
      Meal result = apiInstance.deleteMeal(id);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling MealsApi#deleteMeal");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**UUID**](.md)| Id of the meal |

### Return type

[**Meal**](Meal.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**401** | Authentication information is missing or invalid |  * WWW_Authenticate -  <br>  |
**400** | Invalid Id Supplied |  -  |
**404** | Meal not found |  -  |

<a name="getMealById"></a>
# **getMealById**
> Meal getMealById(id)

Get a meal by its id

Get a meal by id description

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.models.*;
import org.openapitools.client.api.MealsApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost:8080");

    MealsApi apiInstance = new MealsApi(defaultClient);
    UUID id = new UUID(); // UUID | Id of the meal
    try {
      Meal result = apiInstance.getMealById(id);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling MealsApi#getMealById");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**UUID**](.md)| Id of the meal |

### Return type

[**Meal**](Meal.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Found the meal |  -  |
**400** | Invalid Id Supplied |  -  |
**404** | Meal not found |  -  |

<a name="getMeals"></a>
# **getMeals**
> List&lt;Meal&gt; getMeals()

Retrieve all meals

Find all meals

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.models.*;
import org.openapitools.client.api.MealsApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost:8080");

    MealsApi apiInstance = new MealsApi(defaultClient);
    try {
      List<Meal> result = apiInstance.getMeals();
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling MealsApi#getMeals");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**List&lt;Meal&gt;**](Meal.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**404** | No Meals found |  -  |

<a name="updateMeal"></a>
# **updateMeal**
> Meal updateMeal(id, mealUpdateRequest)

Update existing meal

Update existing meal

### Example
```java
// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.models.*;
import org.openapitools.client.api.MealsApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://localhost:8080");

    MealsApi apiInstance = new MealsApi(defaultClient);
    UUID id = new UUID(); // UUID | Id of the meal
    MealUpdateRequest mealUpdateRequest = new MealUpdateRequest(); // MealUpdateRequest | 
    try {
      Meal result = apiInstance.updateMeal(id, mealUpdateRequest);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling MealsApi#updateMeal");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **id** | [**UUID**](.md)| Id of the meal |
 **mealUpdateRequest** | [**MealUpdateRequest**](MealUpdateRequest.md)|  |

### Return type

[**Meal**](Meal.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | OK |  -  |
**400** | Invalid Id Supplied |  -  |
**404** | Meal not found |  -  |

