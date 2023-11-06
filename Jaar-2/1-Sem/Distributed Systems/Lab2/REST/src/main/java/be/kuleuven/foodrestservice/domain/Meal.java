package be.kuleuven.foodrestservice.domain;

import java.util.Objects;
import io.swagger.v3.oas.annotations.media.Schema;

@Schema(name="Meal", description="A Delicious meal")
public class Meal {

    @Schema(name="id", description="Unique id of the meal", format="uuid")
    protected String id;
    @Schema(name="name", description="The name of the meal")
    protected String name;
    @Schema(name="kcal", description="The energetic value of the meal")
    protected Integer kcal;
    @Schema(name="price", description="The price of the meal")
    protected Double price;
    @Schema(name="description", description="A description of the meal")
    protected String description;
    @Schema(name="mealType", description="The type of meal")
    protected MealType mealType;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getKcal() {
        return kcal;
    }

    public void setKcal(Integer kcal) {
        this.kcal = kcal;
    }

    public Double getPrice() {
        return price;
    }

    public void setPrice(Double price) {
        this.price = price;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public MealType getMealType() {
        return mealType;
    }

    public void setMealType(MealType mealType) {
        this.mealType = mealType;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Meal meal = (Meal) o;
        return Objects.equals(id, meal.id) &&
                Objects.equals(name, meal.name) &&
                Objects.equals(kcal, meal.kcal) &&
                Objects.equals(price, meal.price) &&
                Objects.equals(description, meal.description) &&
                mealType == meal.mealType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, name, kcal, price, description, mealType);
    }
}

