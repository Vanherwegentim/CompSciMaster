package be.kuleuven.distributedsystems.cloud.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class Controller {
    @PostMapping("/subscription")
    public void subscription(@RequestBody String body){

    }
}
