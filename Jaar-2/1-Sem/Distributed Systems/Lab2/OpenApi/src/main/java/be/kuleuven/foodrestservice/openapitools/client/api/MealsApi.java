/*
 * Resto
 * Delicious Meal API
 *
 * The version of the OpenAPI document: v1.0.0
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


package be.kuleuven.foodrestservice.openapitools.client.api;

import be.kuleuven.foodrestservice.openapitools.client.ApiCallback;
import be.kuleuven.foodrestservice.openapitools.client.ApiClient;
import be.kuleuven.foodrestservice.openapitools.client.ApiException;
import be.kuleuven.foodrestservice.openapitools.client.ApiResponse;
import be.kuleuven.foodrestservice.openapitools.client.Configuration;
import be.kuleuven.foodrestservice.openapitools.client.Pair;
import be.kuleuven.foodrestservice.openapitools.client.ProgressRequestBody;
import be.kuleuven.foodrestservice.openapitools.client.ProgressResponseBody;

import com.google.gson.reflect.TypeToken;

import java.io.IOException;


import be.kuleuven.foodrestservice.openapitools.client.model.Meal;
import be.kuleuven.foodrestservice.openapitools.client.model.MealUpdateRequest;
import java.util.UUID;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MealsApi {
    private ApiClient localVarApiClient;

    public MealsApi() {
        this(Configuration.getDefaultApiClient());
    }

    public MealsApi(ApiClient apiClient) {
        this.localVarApiClient = apiClient;
    }

    public ApiClient getApiClient() {
        return localVarApiClient;
    }

    public void setApiClient(ApiClient apiClient) {
        this.localVarApiClient = apiClient;
    }

    /**
     * Build call for addMeal
     * @param mealUpdateRequest  (required)
     * @param _callback Callback for upload/download progress
     * @return Call to execute
     * @throws ApiException If fail to serialize the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 201 </td><td> New Meal created </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call addMealCall(MealUpdateRequest mealUpdateRequest, final ApiCallback _callback) throws ApiException {
        Object localVarPostBody = mealUpdateRequest;

        // create path and map variables
        String localVarPath = "/meals";

        List<Pair> localVarQueryParams = new ArrayList<Pair>();
        List<Pair> localVarCollectionQueryParams = new ArrayList<Pair>();
        Map<String, String> localVarHeaderParams = new HashMap<String, String>();
        Map<String, String> localVarCookieParams = new HashMap<String, String>();
        Map<String, Object> localVarFormParams = new HashMap<String, Object>();

        final String[] localVarAccepts = {
            "application/json"
        };
        final String localVarAccept = localVarApiClient.selectHeaderAccept(localVarAccepts);
        if (localVarAccept != null) {
            localVarHeaderParams.put("Accept", localVarAccept);
        }

        final String[] localVarContentTypes = {
            "application/json"
        };
        final String localVarContentType = localVarApiClient.selectHeaderContentType(localVarContentTypes);
        localVarHeaderParams.put("Content-Type", localVarContentType);

        String[] localVarAuthNames = new String[] {  };
        return localVarApiClient.buildCall(localVarPath, "POST", localVarQueryParams, localVarCollectionQueryParams, localVarPostBody, localVarHeaderParams, localVarCookieParams, localVarFormParams, localVarAuthNames, _callback);
    }

    @SuppressWarnings("rawtypes")
    private okhttp3.Call addMealValidateBeforeCall(MealUpdateRequest mealUpdateRequest, final ApiCallback _callback) throws ApiException {
        
        // verify the required parameter 'mealUpdateRequest' is set
        if (mealUpdateRequest == null) {
            throw new ApiException("Missing the required parameter 'mealUpdateRequest' when calling addMeal(Async)");
        }
        

        okhttp3.Call localVarCall = addMealCall(mealUpdateRequest, _callback);
        return localVarCall;

    }

    /**
     * Add a new meal
     * Add a new meal
     * @param mealUpdateRequest  (required)
     * @return Object
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 201 </td><td> New Meal created </td><td>  -  </td></tr>
     </table>
     */
    public Object addMeal(MealUpdateRequest mealUpdateRequest) throws ApiException {
        ApiResponse<Object> localVarResp = addMealWithHttpInfo(mealUpdateRequest);
        return localVarResp.getData();
    }

    /**
     * Add a new meal
     * Add a new meal
     * @param mealUpdateRequest  (required)
     * @return ApiResponse&lt;Object&gt;
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 201 </td><td> New Meal created </td><td>  -  </td></tr>
     </table>
     */
    public ApiResponse<Object> addMealWithHttpInfo(MealUpdateRequest mealUpdateRequest) throws ApiException {
        okhttp3.Call localVarCall = addMealValidateBeforeCall(mealUpdateRequest, null);
        Type localVarReturnType = new TypeToken<Object>(){}.getType();
        return localVarApiClient.execute(localVarCall, localVarReturnType);
    }

    /**
     * Add a new meal (asynchronously)
     * Add a new meal
     * @param mealUpdateRequest  (required)
     * @param _callback The callback to be executed when the API call finishes
     * @return The request call
     * @throws ApiException If fail to process the API call, e.g. serializing the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 201 </td><td> New Meal created </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call addMealAsync(MealUpdateRequest mealUpdateRequest, final ApiCallback<Object> _callback) throws ApiException {

        okhttp3.Call localVarCall = addMealValidateBeforeCall(mealUpdateRequest, _callback);
        Type localVarReturnType = new TypeToken<Object>(){}.getType();
        localVarApiClient.executeAsync(localVarCall, localVarReturnType, _callback);
        return localVarCall;
    }
    /**
     * Build call for deleteMeal
     * @param id Id of the meal (required)
     * @param _callback Callback for upload/download progress
     * @return Call to execute
     * @throws ApiException If fail to serialize the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 401 </td><td> Authentication information is missing or invalid </td><td>  * WWW_Authenticate -  <br>  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call deleteMealCall(UUID id, final ApiCallback _callback) throws ApiException {
        Object localVarPostBody = null;

        // create path and map variables
        String localVarPath = "/meals/{id}"
            .replaceAll("\\{" + "id" + "\\}", localVarApiClient.escapeString(id.toString()));

        List<Pair> localVarQueryParams = new ArrayList<Pair>();
        List<Pair> localVarCollectionQueryParams = new ArrayList<Pair>();
        Map<String, String> localVarHeaderParams = new HashMap<String, String>();
        Map<String, String> localVarCookieParams = new HashMap<String, String>();
        Map<String, Object> localVarFormParams = new HashMap<String, Object>();

        final String[] localVarAccepts = {
            "application/json"
        };
        final String localVarAccept = localVarApiClient.selectHeaderAccept(localVarAccepts);
        if (localVarAccept != null) {
            localVarHeaderParams.put("Accept", localVarAccept);
        }

        final String[] localVarContentTypes = {
            
        };
        final String localVarContentType = localVarApiClient.selectHeaderContentType(localVarContentTypes);
        localVarHeaderParams.put("Content-Type", localVarContentType);

        String[] localVarAuthNames = new String[] {  };
        return localVarApiClient.buildCall(localVarPath, "DELETE", localVarQueryParams, localVarCollectionQueryParams, localVarPostBody, localVarHeaderParams, localVarCookieParams, localVarFormParams, localVarAuthNames, _callback);
    }

    @SuppressWarnings("rawtypes")
    private okhttp3.Call deleteMealValidateBeforeCall(UUID id, final ApiCallback _callback) throws ApiException {
        
        // verify the required parameter 'id' is set
        if (id == null) {
            throw new ApiException("Missing the required parameter 'id' when calling deleteMeal(Async)");
        }
        

        okhttp3.Call localVarCall = deleteMealCall(id, _callback);
        return localVarCall;

    }

    /**
     * Remove a meal
     * Remove an existing meal
     * @param id Id of the meal (required)
     * @return Meal
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 401 </td><td> Authentication information is missing or invalid </td><td>  * WWW_Authenticate -  <br>  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public Meal deleteMeal(UUID id) throws ApiException {
        ApiResponse<Meal> localVarResp = deleteMealWithHttpInfo(id);
        return localVarResp.getData();
    }

    /**
     * Remove a meal
     * Remove an existing meal
     * @param id Id of the meal (required)
     * @return ApiResponse&lt;Meal&gt;
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 401 </td><td> Authentication information is missing or invalid </td><td>  * WWW_Authenticate -  <br>  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public ApiResponse<Meal> deleteMealWithHttpInfo(UUID id) throws ApiException {
        okhttp3.Call localVarCall = deleteMealValidateBeforeCall(id, null);
        Type localVarReturnType = new TypeToken<Meal>(){}.getType();
        return localVarApiClient.execute(localVarCall, localVarReturnType);
    }

    /**
     * Remove a meal (asynchronously)
     * Remove an existing meal
     * @param id Id of the meal (required)
     * @param _callback The callback to be executed when the API call finishes
     * @return The request call
     * @throws ApiException If fail to process the API call, e.g. serializing the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 401 </td><td> Authentication information is missing or invalid </td><td>  * WWW_Authenticate -  <br>  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call deleteMealAsync(UUID id, final ApiCallback<Meal> _callback) throws ApiException {

        okhttp3.Call localVarCall = deleteMealValidateBeforeCall(id, _callback);
        Type localVarReturnType = new TypeToken<Meal>(){}.getType();
        localVarApiClient.executeAsync(localVarCall, localVarReturnType, _callback);
        return localVarCall;
    }
    /**
     * Build call for getMealById
     * @param id Id of the meal (required)
     * @param _callback Callback for upload/download progress
     * @return Call to execute
     * @throws ApiException If fail to serialize the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> Found the meal </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call getMealByIdCall(UUID id, final ApiCallback _callback) throws ApiException {
        Object localVarPostBody = null;

        // create path and map variables
        String localVarPath = "/meals/{id}"
            .replaceAll("\\{" + "id" + "\\}", localVarApiClient.escapeString(id.toString()));

        List<Pair> localVarQueryParams = new ArrayList<Pair>();
        List<Pair> localVarCollectionQueryParams = new ArrayList<Pair>();
        Map<String, String> localVarHeaderParams = new HashMap<String, String>();
        Map<String, String> localVarCookieParams = new HashMap<String, String>();
        Map<String, Object> localVarFormParams = new HashMap<String, Object>();

        final String[] localVarAccepts = {
            "application/json"
        };
        final String localVarAccept = localVarApiClient.selectHeaderAccept(localVarAccepts);
        if (localVarAccept != null) {
            localVarHeaderParams.put("Accept", localVarAccept);
        }

        final String[] localVarContentTypes = {
            
        };
        final String localVarContentType = localVarApiClient.selectHeaderContentType(localVarContentTypes);
        localVarHeaderParams.put("Content-Type", localVarContentType);

        String[] localVarAuthNames = new String[] {  };
        return localVarApiClient.buildCall(localVarPath, "GET", localVarQueryParams, localVarCollectionQueryParams, localVarPostBody, localVarHeaderParams, localVarCookieParams, localVarFormParams, localVarAuthNames, _callback);
    }

    @SuppressWarnings("rawtypes")
    private okhttp3.Call getMealByIdValidateBeforeCall(UUID id, final ApiCallback _callback) throws ApiException {
        
        // verify the required parameter 'id' is set
        if (id == null) {
            throw new ApiException("Missing the required parameter 'id' when calling getMealById(Async)");
        }
        

        okhttp3.Call localVarCall = getMealByIdCall(id, _callback);
        return localVarCall;

    }

    /**
     * Get a meal by its id
     * Get a meal by id description
     * @param id Id of the meal (required)
     * @return Meal
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> Found the meal </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public Meal getMealById(UUID id) throws ApiException {
        ApiResponse<Meal> localVarResp = getMealByIdWithHttpInfo(id);
        return localVarResp.getData();
    }

    /**
     * Get a meal by its id
     * Get a meal by id description
     * @param id Id of the meal (required)
     * @return ApiResponse&lt;Meal&gt;
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> Found the meal </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public ApiResponse<Meal> getMealByIdWithHttpInfo(UUID id) throws ApiException {
        okhttp3.Call localVarCall = getMealByIdValidateBeforeCall(id, null);
        Type localVarReturnType = new TypeToken<Meal>(){}.getType();
        return localVarApiClient.execute(localVarCall, localVarReturnType);
    }

    /**
     * Get a meal by its id (asynchronously)
     * Get a meal by id description
     * @param id Id of the meal (required)
     * @param _callback The callback to be executed when the API call finishes
     * @return The request call
     * @throws ApiException If fail to process the API call, e.g. serializing the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> Found the meal </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call getMealByIdAsync(UUID id, final ApiCallback<Meal> _callback) throws ApiException {

        okhttp3.Call localVarCall = getMealByIdValidateBeforeCall(id, _callback);
        Type localVarReturnType = new TypeToken<Meal>(){}.getType();
        localVarApiClient.executeAsync(localVarCall, localVarReturnType, _callback);
        return localVarCall;
    }
    /**
     * Build call for getMeals
     * @param _callback Callback for upload/download progress
     * @return Call to execute
     * @throws ApiException If fail to serialize the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> No Meals found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call getMealsCall(final ApiCallback _callback) throws ApiException {
        Object localVarPostBody = null;

        // create path and map variables
        String localVarPath = "/meals";

        List<Pair> localVarQueryParams = new ArrayList<Pair>();
        List<Pair> localVarCollectionQueryParams = new ArrayList<Pair>();
        Map<String, String> localVarHeaderParams = new HashMap<String, String>();
        Map<String, String> localVarCookieParams = new HashMap<String, String>();
        Map<String, Object> localVarFormParams = new HashMap<String, Object>();

        final String[] localVarAccepts = {
            "application/json"
        };
        final String localVarAccept = localVarApiClient.selectHeaderAccept(localVarAccepts);
        if (localVarAccept != null) {
            localVarHeaderParams.put("Accept", localVarAccept);
        }

        final String[] localVarContentTypes = {
            
        };
        final String localVarContentType = localVarApiClient.selectHeaderContentType(localVarContentTypes);
        localVarHeaderParams.put("Content-Type", localVarContentType);

        String[] localVarAuthNames = new String[] {  };
        return localVarApiClient.buildCall(localVarPath, "GET", localVarQueryParams, localVarCollectionQueryParams, localVarPostBody, localVarHeaderParams, localVarCookieParams, localVarFormParams, localVarAuthNames, _callback);
    }

    @SuppressWarnings("rawtypes")
    private okhttp3.Call getMealsValidateBeforeCall(final ApiCallback _callback) throws ApiException {
        

        okhttp3.Call localVarCall = getMealsCall(_callback);
        return localVarCall;

    }

    /**
     * Retrieve all meals
     * Find all meals
     * @return List&lt;Meal&gt;
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> No Meals found </td><td>  -  </td></tr>
     </table>
     */
    public List<Meal> getMeals() throws ApiException {
        ApiResponse<List<Meal>> localVarResp = getMealsWithHttpInfo();
        return localVarResp.getData();
    }

    /**
     * Retrieve all meals
     * Find all meals
     * @return ApiResponse&lt;List&lt;Meal&gt;&gt;
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> No Meals found </td><td>  -  </td></tr>
     </table>
     */
    public ApiResponse<List<Meal>> getMealsWithHttpInfo() throws ApiException {
        okhttp3.Call localVarCall = getMealsValidateBeforeCall(null);
        Type localVarReturnType = new TypeToken<List<Meal>>(){}.getType();
        return localVarApiClient.execute(localVarCall, localVarReturnType);
    }

    /**
     * Retrieve all meals (asynchronously)
     * Find all meals
     * @param _callback The callback to be executed when the API call finishes
     * @return The request call
     * @throws ApiException If fail to process the API call, e.g. serializing the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> No Meals found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call getMealsAsync(final ApiCallback<List<Meal>> _callback) throws ApiException {

        okhttp3.Call localVarCall = getMealsValidateBeforeCall(_callback);
        Type localVarReturnType = new TypeToken<List<Meal>>(){}.getType();
        localVarApiClient.executeAsync(localVarCall, localVarReturnType, _callback);
        return localVarCall;
    }
    /**
     * Build call for updateMeal
     * @param id Id of the meal (required)
     * @param mealUpdateRequest  (required)
     * @param _callback Callback for upload/download progress
     * @return Call to execute
     * @throws ApiException If fail to serialize the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call updateMealCall(UUID id, MealUpdateRequest mealUpdateRequest, final ApiCallback _callback) throws ApiException {
        Object localVarPostBody = mealUpdateRequest;

        // create path and map variables
        String localVarPath = "/meals/{id}"
            .replaceAll("\\{" + "id" + "\\}", localVarApiClient.escapeString(id.toString()));

        List<Pair> localVarQueryParams = new ArrayList<Pair>();
        List<Pair> localVarCollectionQueryParams = new ArrayList<Pair>();
        Map<String, String> localVarHeaderParams = new HashMap<String, String>();
        Map<String, String> localVarCookieParams = new HashMap<String, String>();
        Map<String, Object> localVarFormParams = new HashMap<String, Object>();

        final String[] localVarAccepts = {
            "application/json"
        };
        final String localVarAccept = localVarApiClient.selectHeaderAccept(localVarAccepts);
        if (localVarAccept != null) {
            localVarHeaderParams.put("Accept", localVarAccept);
        }

        final String[] localVarContentTypes = {
            "application/json"
        };
        final String localVarContentType = localVarApiClient.selectHeaderContentType(localVarContentTypes);
        localVarHeaderParams.put("Content-Type", localVarContentType);

        String[] localVarAuthNames = new String[] {  };
        return localVarApiClient.buildCall(localVarPath, "PUT", localVarQueryParams, localVarCollectionQueryParams, localVarPostBody, localVarHeaderParams, localVarCookieParams, localVarFormParams, localVarAuthNames, _callback);
    }

    @SuppressWarnings("rawtypes")
    private okhttp3.Call updateMealValidateBeforeCall(UUID id, MealUpdateRequest mealUpdateRequest, final ApiCallback _callback) throws ApiException {
        
        // verify the required parameter 'id' is set
        if (id == null) {
            throw new ApiException("Missing the required parameter 'id' when calling updateMeal(Async)");
        }
        
        // verify the required parameter 'mealUpdateRequest' is set
        if (mealUpdateRequest == null) {
            throw new ApiException("Missing the required parameter 'mealUpdateRequest' when calling updateMeal(Async)");
        }
        

        okhttp3.Call localVarCall = updateMealCall(id, mealUpdateRequest, _callback);
        return localVarCall;

    }

    /**
     * Update existing meal
     * Update existing meal
     * @param id Id of the meal (required)
     * @param mealUpdateRequest  (required)
     * @return Meal
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public Meal updateMeal(UUID id, MealUpdateRequest mealUpdateRequest) throws ApiException {
        ApiResponse<Meal> localVarResp = updateMealWithHttpInfo(id, mealUpdateRequest);
        return localVarResp.getData();
    }

    /**
     * Update existing meal
     * Update existing meal
     * @param id Id of the meal (required)
     * @param mealUpdateRequest  (required)
     * @return ApiResponse&lt;Meal&gt;
     * @throws ApiException If fail to call the API, e.g. server error or cannot deserialize the response body
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public ApiResponse<Meal> updateMealWithHttpInfo(UUID id, MealUpdateRequest mealUpdateRequest) throws ApiException {
        okhttp3.Call localVarCall = updateMealValidateBeforeCall(id, mealUpdateRequest, null);
        Type localVarReturnType = new TypeToken<Meal>(){}.getType();
        return localVarApiClient.execute(localVarCall, localVarReturnType);
    }

    /**
     * Update existing meal (asynchronously)
     * Update existing meal
     * @param id Id of the meal (required)
     * @param mealUpdateRequest  (required)
     * @param _callback The callback to be executed when the API call finishes
     * @return The request call
     * @throws ApiException If fail to process the API call, e.g. serializing the request body object
     * @http.response.details
     <table summary="Response Details" border="1">
        <tr><td> Status Code </td><td> Description </td><td> Response Headers </td></tr>
        <tr><td> 200 </td><td> OK </td><td>  -  </td></tr>
        <tr><td> 400 </td><td> Invalid Id Supplied </td><td>  -  </td></tr>
        <tr><td> 404 </td><td> Meal not found </td><td>  -  </td></tr>
     </table>
     */
    public okhttp3.Call updateMealAsync(UUID id, MealUpdateRequest mealUpdateRequest, final ApiCallback<Meal> _callback) throws ApiException {

        okhttp3.Call localVarCall = updateMealValidateBeforeCall(id, mealUpdateRequest, _callback);
        Type localVarReturnType = new TypeToken<Meal>(){}.getType();
        localVarApiClient.executeAsync(localVarCall, localVarReturnType, _callback);
        return localVarCall;
    }
}