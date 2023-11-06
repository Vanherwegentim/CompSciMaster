package be.kuleuven.foodrestservice.controllers;

import be.kuleuven.foodrestservice.domain.Meal;
import be.kuleuven.foodrestservice.domain.MealsRepository;
import be.kuleuven.foodrestservice.exceptions.MealNotFoundException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.hateoas.EntityModel;
import org.springframework.web.bind.annotation.*;

import java.util.Collection;
import java.util.Optional;

@RestController
public class MealsRestRpcStyleController {

    private final MealsRepository mealsRepository;

    @Autowired
    MealsRestRpcStyleController(MealsRepository mealsRepository) {
        this.mealsRepository = mealsRepository;
    }

    @GetMapping("/restrpc/meals/{id}")
    Meal getMealById(@PathVariable String id) {
        Optional<Meal> meal = mealsRepository.findMeal(id);

        return meal.orElseThrow(() -> new MealNotFoundException(id));
    }

    @GetMapping("/restrpc/meals")
    Collection<Meal> getMeals() {
        return mealsRepository.getAllMeal();
    }

    @GetMapping("/restrpc/largestMeal")
    Meal getLargestMeal() {
        Optional<Meal> meal = mealsRepository.findLargestMeal();

        return meal.orElseThrow(() -> new MealNotFoundException("No largest meal found"));
    }
    @GetMapping("/restrpc/cheapestMeal")
    Meal getCheapestMeal() {
        Optional<Meal> meal = mealsRepository.findCheapestMeal();

        return meal.orElseThrow(() -> new MealNotFoundException("No cheapest meal found"));
    }
}