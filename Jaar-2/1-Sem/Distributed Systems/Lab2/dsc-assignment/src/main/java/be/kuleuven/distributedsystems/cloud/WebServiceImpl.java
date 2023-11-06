package be.kuleuven.distributedsystems.cloud;

import be.kuleuven.distributedsystems.cloud.entities.Flight;
import be.kuleuven.distributedsystems.cloud.entities.Seat;
import be.kuleuven.distributedsystems.cloud.entities.Ticket;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.hateoas.CollectionModel;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.*;

import static java.util.stream.Collectors.groupingBy;

@Service
public class WebServiceImpl implements WebService {

    @Autowired
    private final WebClient.Builder webClientBuilder;

    private static final ObjectMapper mapper = new ObjectMapper();
    private final static String key = "Iw8zeveVyaPNWonPNaU0213uw3g6Ei";

    private final PubSub pubSub;

    List<WebClient> airlines;

    public WebServiceImpl(WebClient.Builder webClientBuilder) {
        this.webClientBuilder = webClientBuilder;
        airlines = new ArrayList<WebClient>();
        airlines.add(this.webClientBuilder.baseUrl("https://reliable-airline.com").build());
        airlines.add(this.webClientBuilder.baseUrl("https://unreliable-airline.com").build());
        pubSub = new PubSub();

    }


    public Flight[] getFlights() {
        Flight[] allFlights = new Flight[0];

        pubSub.publishMessage("getFlights");

        Iterator itr = airlines.iterator();
        while (itr.hasNext()) {

            WebClient x = (WebClient) itr.next();
            var result = x
                    .get()
                    .uri(uriBuilder -> uriBuilder
                            .path("/flights")
                            .queryParam("key", key)
                            .build())
                    .retrieve()
                    .bodyToMono(new ParameterizedTypeReference<CollectionModel<Flight>>() {
                    }).log()
                    .retry(3)
                    .block()
                    .getContent();
            Flight[] flights = result.toArray(new Flight[result.size()]);
            Flight[] both = Arrays.copyOf(allFlights, allFlights.length + flights.length);
            System.arraycopy(flights, 0, both, allFlights.length, flights.length);
            allFlights = both;
        }

        return allFlights;
    }


    public Flight getFlight(String airline, String flightId) {
        Mono<Flight> response = webClientBuilder.baseUrl("https://" + airline).build()
                .get()
                .uri(uriBuilder -> uriBuilder
                        .path("/flights/" + flightId)
                        .queryParam("key", key)
                        .build())
                .retrieve()
                .bodyToMono(Flight.class).log().retry(3);
        return response.block();
    }

    public String[] getFlightTimes(String airline, String flightId) {
        var times = webClientBuilder.baseUrl("https://" + airline).build()
                .get()
                .uri(uriBuilder -> uriBuilder
                        .path("/flights/" + flightId + "/times")
                        .queryParam("key", key)
                        .build())
                .retrieve()
                .bodyToMono(new ParameterizedTypeReference<CollectionModel<String>>() {
                }).log()
                .retry(3)
                .block()
                .getContent();

        return Arrays.stream(times.toArray(new String[0])).sorted().toArray(String[]::new);
    }

    public Map<String, List<Seat>> getAvailableSeats(String airline, String flightId, String time) {
        var seats = webClientBuilder.baseUrl("https://" + airline).build()
                .get()
                .uri(uriBuilder -> uriBuilder
                        .path("/flights/" + flightId + "/seats")
                        .queryParam("time", time)
                        .queryParam("available", true)
                        .queryParam("key", key)
                        .build())
                .retrieve()
                .bodyToMono(new ParameterizedTypeReference<CollectionModel<Seat>>() {
                }).log()
                .retry(3)
                .block()
                .getContent();
        Seat[] zitjes = seats.toArray(new Seat[0]);

        return Arrays.stream(zitjes).sorted(Comparator.comparing(Seat::getName)).collect(groupingBy(Seat::getType));

    }

    public Seat getSeat(String airline, String flightId, String seatId) {
        return webClientBuilder.baseUrl("https://" + airline).build()
                .get()
                .uri(uriBuilder -> uriBuilder
                        .path("/flights/" + flightId + "/seats/" + seatId)
                        .queryParam("key", key)
                        .build())
                .retrieve()
                .bodyToMono(Seat.class).log().retry(3).block();
    }

    public Ticket putSeat(String airline, String flightId, String seatId, String user, String bookingReference) {
        return webClientBuilder.baseUrl("https://" + airline).build()
                .put()
                .uri(uriBuilder -> uriBuilder
                        .path("/flights/" + flightId + "/seats/" + seatId + "/ticket")
                        .queryParam("customer", user)
                        .queryParam("bookingReference", bookingReference)
                        .queryParam("key", key)
                        .build())
                .retrieve().bodyToMono(Ticket.class).retry(3).block();
    }

    @Override
    public void cancelTicket(String airline, String flightId, String seatId, String ticketId) {
        webClientBuilder.baseUrl("https://" + airline).build()
                .delete()
                .uri(uriBuilder -> uriBuilder
                        .path("/flights/" + flightId + "/seats/" + seatId + "/ticket/" + ticketId)
                        .queryParam("key", key)
                        .build())
                .retrieve().bodyToMono(Ticket.class).retry(3).block();
    }


}
