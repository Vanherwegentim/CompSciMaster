package be.kuleuven.distributedsystems.cloud.entities;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class Ticket {
    private String airline;
    private String flightId;
    private String seatId;
    private String ticketId;
    private String customer;
    private String bookingReference;

    public Ticket() {
    }

    public Ticket(String airline, String flightId, String seatId, String ticketId, String customer, String bookingReference) {
        this.airline = airline;
        this.flightId = flightId;
        this.seatId = seatId;
        this.ticketId = ticketId;
        this.customer = customer;
        this.bookingReference = bookingReference;
    }

    public String getAirline() {
        return airline;
    }

    public String getFlightId() {
        return flightId;
    }

    public String getSeatId() {
        return this.seatId;
    }

    public String getTicketId() {
        return this.ticketId;
    }

    public String getCustomer() {
        return this.customer;
    }

    public String getBookingReference() {
        return this.bookingReference;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Ticket other)) {
            return false;
        }
        return this.ticketId.equals(other.ticketId)
                && this.seatId.equals(other.seatId)
                && this.flightId.equals(other.flightId)
                && this.airline.equals(other.airline);
    }

    @Override
    public int hashCode() {
        return this.airline.hashCode() * this.flightId.hashCode() * this.seatId.hashCode() * this.ticketId.hashCode();
    }
}
