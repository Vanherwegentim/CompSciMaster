package be.kuleuven.foodrestservice.controller;

import be.kuleuven.foodrestservice.openapitools.client.api.MealsApi;
import be.kuleuven.foodrestservice.domain.MealsRepository;
import be.kuleuven.foodrestservice.exceptions.MealNotFoundException;
import be.kuleuven.foodrestservice.openapitools.client.model.*;

import java.util.*;

import org.springframework.http.ResponseEntity;

import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

@RestController
public class MealsController implements MealsApi {

    private static final MealsRepository mealsRepository = new MealsRepository();

    @Override
    public ResponseEntity<List<Meal>> getMeals() {
        return ResponseEntity.ok(mealsRepository.getAllMeals());
    }

    @Override
    public ResponseEntity<Object> addMeal(MealUpdateRequest mealUpdateRequest) {
        Meal newMeal = mealsRepository.addMeal(mealUpdateRequest);

        // In case of creation a Link is returned in the headers pointing to the new entry
        return ResponseEntity.created(ServletUriComponentsBuilder.fromCurrentRequest().path(newMeal.getId().toString()).build().toUri()).body(newMeal);
    }

    @Override
    public ResponseEntity<Meal> deleteMeal(UUID id) {
        Meal meal = mealsRepository.deleteMeal(id).orElseThrow(() -> new MealNotFoundException(id.toString()));
        return ResponseEntity.ok(meal);

    }

    @Override
    public ResponseEntity<Meal> getMealById(UUID id) {
        Meal meal = mealsRepository.findMeal(id).orElseThrow(() -> new MealNotFoundException(id.toString()));
        return ResponseEntity.ok(meal);

    }

    @Override
    public ResponseEntity<Meal> updateMeal(UUID id, MealUpdateRequest mealUpdateRequest) {
        Meal updatedMeal = mealsRepository.updateMeal(id, mealUpdateRequest);
        return ResponseEntity.ok(updatedMeal);
    }
}
