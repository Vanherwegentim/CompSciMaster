package be.kuleuven.distributedsystems.cloud.controller;

import be.kuleuven.distributedsystems.cloud.WebService;
import be.kuleuven.distributedsystems.cloud.entities.Booking;
import be.kuleuven.distributedsystems.cloud.entities.Flight;
import be.kuleuven.distributedsystems.cloud.entities.Seat;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpMethod;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

import java.sql.Time;
import java.util.Collections;
import java.util.List;
import java.util.Map;


@RestController
@RequestMapping("api")
public class FlightController {
    @Autowired
    private WebService webService;

    private static final String apiKey = "Iw8zeveVyaPNWonPNaU0213uw3g6Ei";

    @GetMapping("getFlights")
    public Flight[] getFlights(){
        return webService.getFlights();
    }

    @GetMapping("getFlight")
    public Flight getFlights(String airline, String flightId){
        return webService.getFlight(airline, flightId);
    }

    @GetMapping("getFlightTimes")
    public String[] getFlightTimes(String airline, String flightId){
        return webService.getFlightTimes(airline, flightId);
    }

    @GetMapping("getAvailableSeats")
    public Map<String, List<Seat>> getAvailableSeats(String airline, String flightId, String time){
        return webService.getAvailableSeats(airline, flightId, time);
    }

    @GetMapping("getSeat")
    public Seat getSeat(String airline, String flightId, String seatId){
        return webService.getSeat(airline, flightId, seatId);
    }
}
