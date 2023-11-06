package be.kuleuven.distributedsystems.cloud.entities;

import java.io.Serializable;
import java.util.UUID;

public class Quote {

    private String airline;
    private String flightId;
    private String seatId;

    public Quote() {
    }

    public Quote(String airline, String flightId, String seatId) {
        this.airline = airline;
        this.flightId = flightId;
        this.seatId = seatId;
    }

    public String getAirline() {
        return airline;
    }

    public void setAirline(String airline) {
        this.airline = airline;
    }

    public String getFlightId() {
        return flightId;
    }

    public void setFlightId(String flightId) {
        this.flightId = flightId;
    }

    public String getSeatId() {
        return this.seatId;
    }

    public void setSeatId(String seatId) {
        this.seatId = seatId;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Quote other)) {
            return false;
        }
        return this.airline.equals(other.airline)
                && this.flightId.equals(other.flightId)
                && this.seatId.equals(other.seatId);
    }

    @Override
    public int hashCode() {
        return this.airline.hashCode() * this.flightId.hashCode() * this.seatId.hashCode();
    }
}
