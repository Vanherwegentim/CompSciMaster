package be.kuleuven.foodrestservice.controllers;

import be.kuleuven.foodrestservice.domain.Meal;
import be.kuleuven.foodrestservice.domain.MealType;
import be.kuleuven.foodrestservice.domain.MealsRepository;
import be.kuleuven.foodrestservice.exceptions.MealNotFoundException;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.hateoas.CollectionModel;
import org.springframework.hateoas.EntityModel;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.support.ServletUriComponentsBuilder;

import java.net.URI;
import java.util.*;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.*;

@RestController
public class MealsRestController {

    private final MealsRepository mealsRepository;

    @Autowired
    MealsRestController(MealsRepository mealsRepository) {
        this.mealsRepository = mealsRepository;
    }

    @Operation(summary = "Get a meal by its id", description = "Get a meal by id description")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "Found the meal",
                    content = {@Content(mediaType = "application/json", schema = @Schema(implementation = Meal.class))}),
            @ApiResponse(responseCode = "404", description = "Meal not found", content = @Content)})

    @GetMapping("/rest/meals/{id}")
    ResponseEntity<?> getMealById(
            @Parameter(description = "Id of the meal", schema = @Schema(format = "uuid", type = "string"))
            @PathVariable String id) {
        Meal meal = mealsRepository.findMeal(id).orElseThrow(() -> new MealNotFoundException(id));
        EntityModel<Meal> mealEntityModel = mealToEntityModel(id, meal);
        return ResponseEntity.ok(mealEntityModel);
    }

    @GetMapping("/rest/meals")
    CollectionModel<EntityModel<Meal>> getMeals() {
        Collection<Meal> meals = mealsRepository.getAllMeal();

        List<EntityModel<Meal>> mealEntityModels = new ArrayList<>();
        for (Meal m : meals) {
            EntityModel<Meal> em = mealToEntityModel(m.getId(), m);
            mealEntityModels.add(em);
        }
        return CollectionModel.of(mealEntityModels,
                linkTo(methodOn(MealsRestController.class).getMeals()).withSelfRel());
    }
    @GetMapping("/rest/meals/largestMeal")
    ResponseEntity<?> getLargestMeal(){
            Meal meal = mealsRepository.findLargestMeal().orElseThrow(() -> new MealNotFoundException("No largest meal was found"));
            EntityModel<Meal> em = mealToEntityModel(meal.getId(),meal);
            return ResponseEntity.ok(em);
    }
    @GetMapping("/rest/meals/cheapestMeal")
    ResponseEntity<?> getCheapestMeal(){
        Meal meal = mealsRepository.findCheapestMeal().orElseThrow(() -> new MealNotFoundException("No largest meal was found"));
        EntityModel<Meal> em = mealToEntityModel(meal.getId(),meal);
        return ResponseEntity.ok(em);
    }

    @PostMapping("rest/meals/addMeal")
    ResponseEntity<String> addNewMeal(@RequestBody Meal meal) {
        if (meal == null) {
            return ResponseEntity.notFound().build();
        } else {
            mealsRepository.addMeal(meal);
            URI location = URI.create(String.format("/rest/meals/%s", meal.getId()));
            String baseUrl = ServletUriComponentsBuilder.fromCurrentContextPath().build().toUriString();

            return ResponseEntity.created(location).body(baseUrl.concat(location.toString()));
        }
    }

    @PutMapping("rest/meals/updateMeal")
    ResponseEntity<String> updateMeal(@RequestBody Meal meal){
        if (meal == null) {
            return ResponseEntity.notFound().build();
        } else {
            mealsRepository.addMeal(meal);
            URI location = URI.create(String.format("/rest/meals/%s", meal.getId()));
            String baseUrl = ServletUriComponentsBuilder.fromCurrentContextPath().build().toUriString();

            return ResponseEntity.created(location).body(baseUrl.concat(location.toString()));
        }
    }

    @DeleteMapping("/rest/meals/{id}")
    ResponseEntity<?> DeleteMealById(
            @Parameter(description = "Id of the meal", schema = @Schema(format = "uuid", type = "string"))
            @PathVariable String id) {
        mealsRepository.deleteMeal(id);
        return ResponseEntity.ok().build();
    }




    private EntityModel<Meal> mealToEntityModel(String id, Meal meal) {
        return EntityModel.of(meal,
                linkTo(methodOn(MealsRestController.class).getMealById(id)).withSelfRel(),
                linkTo(methodOn(MealsRestController.class).getMeals()).withRel("All Meals"));
    }
}
