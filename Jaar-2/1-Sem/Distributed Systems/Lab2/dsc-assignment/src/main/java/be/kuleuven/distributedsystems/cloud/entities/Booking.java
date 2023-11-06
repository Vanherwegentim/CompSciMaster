package be.kuleuven.distributedsystems.cloud.entities;

import com.google.cloud.Timestamp;
import com.google.type.DateTime;

import java.time.LocalDateTime;
import java.util.*;

public class Booking {
    private String id;
    private Date time;
    private List<Ticket> tickets;
    private String customer;

    public Booking(String id, List<Ticket> tickets, Date time, String customer) {
        this.id = id;
        this.tickets = tickets;
        this.time = time;
        this.customer = customer;
    }

    public Booking(){}

    public String getId() {
        return this.id;
    }

    public Date getTime() {
        return this.time;
    }

    public List<Ticket> getTickets() {
        return this.tickets;
    }

    public void setTickets(List<Ticket> tickets) {
        this.tickets = tickets;
    }

    public String getCustomer() {
        return this.customer;
    }
}
