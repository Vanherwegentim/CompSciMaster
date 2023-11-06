package be.kuleuven.distributedsystems.cloud;

import be.kuleuven.distributedsystems.cloud.auth.SecurityFilter;
import be.kuleuven.distributedsystems.cloud.auth.WebSecurityConfig;
import be.kuleuven.distributedsystems.cloud.entities.Booking;
import be.kuleuven.distributedsystems.cloud.entities.Quote;
import be.kuleuven.distributedsystems.cloud.entities.Ticket;
import be.kuleuven.distributedsystems.cloud.entities.User;
import com.google.api.client.util.DateTime;
import com.google.api.core.ApiFunction;
import com.google.api.gax.grpc.InstantiatingGrpcChannelProvider;
import com.google.cloud.Timestamp;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.FirestoreOptions;
import com.google.cloud.firestore.QueryDocumentSnapshot;
import com.google.cloud.firestore.v1.FirestoreClient;
import io.grpc.ManagedChannelBuilder;
import org.springframework.beans.factory.annotation.Autowired;

import javax.annotation.PostConstruct;
import javax.annotation.Resource;
import java.io.IOException;
import java.time.Instant;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ExecutionException;

public class BookingManager {
    List<User> customers;
    List<Booking> bookings;



    private final Firestore getFirestore;




    @PostConstruct
    void constructor(){

    }






    public BookingManager()  {
        this.customers = new ArrayList<User>();
        this.bookings = new ArrayList<Booking>();
        getFirestore = FirestoreOptions.newBuilder()
                .setProjectId("demo-distributed-systems-kul")
                .setChannelProvider(InstantiatingGrpcChannelProvider.newBuilder()
                        .setEndpoint("localhost:8084")
                        .setChannelConfigurator(
                                ManagedChannelBuilder::usePlaintext)
                        .build())
                .setCredentials(new FirestoreOptions.EmulatorCredentials())
                .build().getService();



    }
    public Booking createBooking(Quote[] quotes){
        String bookingId = UUID.randomUUID().toString();

        List<Ticket> tickets = new ArrayList<>(Arrays.stream(quotes)
                .map(q ->
                        new Ticket(q.getAirline(), q.getFlightId(),
                                q.getSeatId(), UUID.randomUUID().toString(),
                                WebSecurityConfig.getUser().getEmail(),
                                bookingId)).toList());

        Booking booking =
                new Booking(bookingId, tickets, Date.from(Instant.now()),
                        WebSecurityConfig.getUser().getEmail());
        return booking;
    }
    public List<User> getCustomers() {
        return customers;
    }

    public void setCustomers(List<User> customers) {
        this.customers = customers;
    }

    public void addCustomer(User user){
        this.customers.add(user);
    }

    public List<Booking> getBookings() {
        List<Booking> bookingList = new ArrayList<>();

        try {
            List<QueryDocumentSnapshot> documents =
                    getFirestore.collection("bookings")
                    .whereEqualTo("customer",
                            WebSecurityConfig.getUser().getEmail())
                            .get().get().getDocuments();
            for (DocumentSnapshot document : documents) {
                try {
                    bookingList.add(document.toObject(Booking.class));
                } catch (Exception e) {
                    System.out.println(e);
                }
            }
        }
        catch (Exception ie){
            System.out.println(ie);
        }

        return  bookingList;
    }

    public List<Booking> getAllBookings() {
        List<Booking> bookingList = new ArrayList<>();
        try {
            List<QueryDocumentSnapshot> documents = getFirestore.collection("bookings").get().get().getDocuments();
            for (DocumentSnapshot document : documents) {
                try {
                    bookingList.add(document.toObject(Booking.class));
                } catch (Exception e) {
                    System.out.println(e);
                }
            }
        }
        catch (Exception ie){
            System.out.println(ie);
        }

        return  bookingList;
    }

    public void setBookings(List<Booking> bookings) {
        this.bookings = bookings;
    }

    public void addBooking(Booking booking) {
        getFirestore.collection("bookings").document(booking.getId()).set(booking);
    }
}

