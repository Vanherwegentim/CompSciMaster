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


package be.kuleuven.foodrestservice.openapitools.client.model;

import java.util.Objects;
import java.util.Arrays;
import com.google.gson.TypeAdapter;
import com.google.gson.annotations.JsonAdapter;
import com.google.gson.annotations.SerializedName;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import java.io.IOException;

/**
 * Request body for a meal
 */
@ApiModel(description = "Request body for a meal")
@javax.annotation.Generated(value = "org.openapitools.codegen.languages.JavaClientCodegen", date = "2023-10-18T17:58:17.039264100+02:00[Europe/Berlin]")
public class MealUpdateRequest {
  public static final String SERIALIZED_NAME_NAME = "name";
  @SerializedName(SERIALIZED_NAME_NAME)
  private String name;

  public static final String SERIALIZED_NAME_PRICE = "price";
  @SerializedName(SERIALIZED_NAME_PRICE)
  private Double price;

  public static final String SERIALIZED_NAME_KCAL = "kcal";
  @SerializedName(SERIALIZED_NAME_KCAL)
  private Integer kcal;

  public static final String SERIALIZED_NAME_DESCRIPTION = "description";
  @SerializedName(SERIALIZED_NAME_DESCRIPTION)
  private String description;

  /**
   * The type of meal
   */
  @JsonAdapter(MealTypeEnum.Adapter.class)
  public enum MealTypeEnum {
    VEGAN("VEGAN"),
    
    VEGGIE("VEGGIE"),
    
    MEAT("MEAT"),
    
    FISH("FISH");

    private String value;

    MealTypeEnum(String value) {
      this.value = value;
    }

    public String getValue() {
      return value;
    }

    @Override
    public String toString() {
      return String.valueOf(value);
    }

    public static MealTypeEnum fromValue(String value) {
      for (MealTypeEnum b : MealTypeEnum.values()) {
        if (b.value.equals(value)) {
          return b;
        }
      }
      throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }

    public static class Adapter extends TypeAdapter<MealTypeEnum> {
      @Override
      public void write(final JsonWriter jsonWriter, final MealTypeEnum enumeration) throws IOException {
        jsonWriter.value(enumeration.getValue());
      }

      @Override
      public MealTypeEnum read(final JsonReader jsonReader) throws IOException {
        String value =  jsonReader.nextString();
        return MealTypeEnum.fromValue(value);
      }
    }
  }

  public static final String SERIALIZED_NAME_MEAL_TYPE = "mealType";
  @SerializedName(SERIALIZED_NAME_MEAL_TYPE)
  private MealTypeEnum mealType;


  public MealUpdateRequest name(String name) {
    
    this.name = name;
    return this;
  }

   /**
   * The name of the meal
   * @return name
  **/
  @ApiModelProperty(required = true, value = "The name of the meal")

  public String getName() {
    return name;
  }


  public void setName(String name) {
    this.name = name;
  }


  public MealUpdateRequest price(Double price) {
    
    this.price = price;
    return this;
  }

   /**
   * The price of the meal
   * @return price
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "The price of the meal")

  public Double getPrice() {
    return price;
  }


  public void setPrice(Double price) {
    this.price = price;
  }


  public MealUpdateRequest kcal(Integer kcal) {
    
    this.kcal = kcal;
    return this;
  }

   /**
   * The energetic value of the meal
   * @return kcal
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "The energetic value of the meal")

  public Integer getKcal() {
    return kcal;
  }


  public void setKcal(Integer kcal) {
    this.kcal = kcal;
  }


  public MealUpdateRequest description(String description) {
    
    this.description = description;
    return this;
  }

   /**
   * A description of the meal
   * @return description
  **/
  @javax.annotation.Nullable
  @ApiModelProperty(value = "A description of the meal")

  public String getDescription() {
    return description;
  }


  public void setDescription(String description) {
    this.description = description;
  }


  public MealUpdateRequest mealType(MealTypeEnum mealType) {
    
    this.mealType = mealType;
    return this;
  }

   /**
   * The type of meal
   * @return mealType
  **/
  @ApiModelProperty(required = true, value = "The type of meal")

  public MealTypeEnum getMealType() {
    return mealType;
  }


  public void setMealType(MealTypeEnum mealType) {
    this.mealType = mealType;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    MealUpdateRequest mealUpdateRequest = (MealUpdateRequest) o;
    return Objects.equals(this.name, mealUpdateRequest.name) &&
        Objects.equals(this.price, mealUpdateRequest.price) &&
        Objects.equals(this.kcal, mealUpdateRequest.kcal) &&
        Objects.equals(this.description, mealUpdateRequest.description) &&
        Objects.equals(this.mealType, mealUpdateRequest.mealType);
  }

  @Override
  public int hashCode() {
    return Objects.hash(name, price, kcal, description, mealType);
  }


  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class MealUpdateRequest {\n");
    sb.append("    name: ").append(toIndentedString(name)).append("\n");
    sb.append("    price: ").append(toIndentedString(price)).append("\n");
    sb.append("    kcal: ").append(toIndentedString(kcal)).append("\n");
    sb.append("    description: ").append(toIndentedString(description)).append("\n");
    sb.append("    mealType: ").append(toIndentedString(mealType)).append("\n");
    sb.append("}");
    return sb.toString();
  }

  /**
   * Convert the given object to string with each line indented by 4 spaces
   * (except the first line).
   */
  private String toIndentedString(Object o) {
    if (o == null) {
      return "null";
    }
    return o.toString().replace("\n", "\n    ");
  }

}
