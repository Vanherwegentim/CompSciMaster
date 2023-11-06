package be.kuleuven.distributedsystems.cloud.controller;

import be.kuleuven.distributedsystems.cloud.BookingManager;
import be.kuleuven.distributedsystems.cloud.WebService;
import be.kuleuven.distributedsystems.cloud.auth.WebSecurityConfig;
import be.kuleuven.distributedsystems.cloud.entities.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import javax.annotation.PostConstruct;
import javax.annotation.Resource;
import java.awt.print.Book;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Stream;


@RestController
@RequestMapping("api")
public class BookingController {
    @Autowired
    private WebService webService;
    private final BookingManager bookingManager = new BookingManager();

    private static final String apiKey = "Iw8zeveVyaPNWonPNaU0213uw3g6Ei";



    @GetMapping("getBookings")
    public Booking[] getBookings(){
        List<Booking> bookings = bookingManager.getBookings();
        return bookings.toArray(Booking[]::new);
    }

    @GetMapping("getAllBookings")
    @PreAuthorize("hasAuthority('manager')")
    public Booking[] getAllBookings(){
        if(!WebSecurityConfig.getUser().isManager()) return null;

        List<Booking> bookings = bookingManager.getAllBookings();
        return bookings.toArray(Booking[]::new);
    }

    @GetMapping("getBestCustomers")
    @PreAuthorize("hasAuthority('manager')")
    public String[] getBestCustomers(){
        if(!WebSecurityConfig.getUser().isManager()) return null;

        List<Booking> bookings = bookingManager.getAllBookings();
        Booking maxBooking = bookings.stream().max(Comparator.comparing(booking -> booking.getTickets().size())).orElseThrow(NoSuchElementException::new);
        Stream<Booking> maxBookings = bookings.stream().filter(booking -> booking.getTickets().size() == maxBooking.getTickets().size());
        return maxBookings.map(Booking::getCustomer).distinct().toArray(String[]::new);
    }

    @PostMapping("confirmQuotes")
    public void confirmQuotes(@RequestBody Quote[] quotes){
        Booking booking = bookingManager.createBooking(quotes);
        List<Ticket> confirmedTickets = new ArrayList<>();
        try{
            for (Quote quote: quotes) {
                Ticket ticket = webService.putSeat(quote.getAirline(),
                        quote.getFlightId(), quote.getSeatId(),
                        WebSecurityConfig.getUser().getEmail() , booking.getId());
                confirmedTickets.add(ticket);
            }
            booking.setTickets(confirmedTickets);
            bookingManager.addBooking(booking);
        }catch(Exception e){
            for(Ticket ticketToCancel: confirmedTickets){
                webService.cancelTicket(ticketToCancel.getAirline(), ticketToCancel.getFlightId(), ticketToCancel.getSeatId(), ticketToCancel.getTicketId());
            }
        }
    }

}
