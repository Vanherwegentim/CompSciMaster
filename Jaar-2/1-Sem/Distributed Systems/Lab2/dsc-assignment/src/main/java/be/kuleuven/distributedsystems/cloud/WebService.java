package be.kuleuven.distributedsystems.cloud;

import be.kuleuven.distributedsystems.cloud.entities.*;

import java.util.List;
import java.util.Map;
import java.util.UUID;

public interface WebService {
    Flight[] getFlights();
    Flight getFlight(String airline, String flightId);

    String[] getFlightTimes(String airline, String flightId);

    Map<String, List<Seat>> getAvailableSeats(String airline, String flightId, String time);

    Seat getSeat(String airline, String flightId, String seatId);

    Ticket putSeat(String airline , String flightId, String seatId, String user, String bookingReference);

    void cancelTicket(String airline, String toString, String toString1, String ticketId);
}
