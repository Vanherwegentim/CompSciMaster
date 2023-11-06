package be.kuleuven.foodrestservice.domain;

import be.kuleuven.foodrestservice.exceptions.MealNotFoundException;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Component;
import org.springframework.util.Assert;
import be.kuleuven.foodrestservice.model.*;

import java.util.*;

@Component
public class MealsRepository {
    // map: id -> meal
    private static final Map<String, Meal> meals = new HashMap<>();

    @PostConstruct
    public void initData() {

        Meal a = new Meal();
        a.setId(UUID.fromString("5268203c-de76-4921-a3e3-439db69c462a"));
        a.setName("Steak");
        a.setDescription("Steak with fries");
        a.setMealType(Meal.MealTypeEnum.MEAT);
        a.setKcal(1100);
        a.setPrice((10.00));

        meals.put(a.getId().toString(), a);

        Meal b = new Meal();
        b.setId(UUID.fromString("4237681a-441f-47fc-a747-8e0169bacea1"));
        b.setName("Portobello");
        b.setDescription("Portobello Mushroom Burger");
        b.setMealType(Meal.MealTypeEnum.VEGAN);
        b.setKcal(637);
        b.setPrice((7.00));

        meals.put(b.getId().toString(), b);

        Meal c = new Meal();
        c.setId(UUID.fromString("cfd1601f-29a0-485d-8d21-7607ec0340c8"));
        c.setName("Fish and Chips");
        c.setDescription("Fried fish with chips");
        c.setMealType(Meal.MealTypeEnum.FISH);
        c.setKcal(950);
        c.setPrice(5.00);

        meals.put(c.getId().toString(), c);
    }

    public Optional<Meal> findMeal(UUID id) {
        Assert.notNull(id, "The meal id must not be null");
        Meal meal = meals.get(id.toString());
        return Optional.ofNullable(meal);
    }

    public List<Meal> getAllMeals() {
        return new LinkedList<>(meals.values());
    }

    public Optional<Meal> deleteMeal(UUID id) {
        Assert.notNull(id, "The meal id must not be null");
        Meal meal = meals.remove(id.toString());
        return Optional.ofNullable(meal);
    }

    public Meal addMeal(MealUpdateRequest mealUpdateRequest) {
        Assert.notNull(mealUpdateRequest, "The meal must not be null");
        Meal meal = createMeal(mealUpdateRequest);
        meal.setId(UUID.randomUUID());
        meals.put(meal.getId().toString(), meal);
        return meal;
    }

    public Meal updateMeal(UUID id, MealUpdateRequest mealRequest) {
        Assert.notNull(id, "The meal id must not be null");
        Assert.notNull(mealRequest, "The meal must not be null");
        // The meal should exist in case of an update
        // You could also opt to create if it doesn't exist. then you have to
        // return status code 201
        findMeal(id).orElseThrow(() -> new MealNotFoundException(id.toString()));
        Meal meal = createMeal(mealRequest);
        meal.setId(id);
        meals.put(id.toString(), meal);
        return meal;
    }

    private Meal createMeal(MealUpdateRequest mealRequest) {
        Meal meal = new Meal();

        meal.setDescription(mealRequest.getDescription());
        meal.setMealType(Meal.MealTypeEnum.fromValue(mealRequest.getMealType().toString()));
        meal.setKcal(mealRequest.getKcal());
        meal.setName(mealRequest.getName());
        meal.setKcal(mealRequest.getKcal());
        meal.setPrice(mealRequest.getPrice());

        return meal;
    }
}
