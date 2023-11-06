package hotel;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.rmi.server.UnicastRemoteObject;

public class BookingManager implements BookingManagerRemote{

	private Room[] rooms;

	public BookingManager() {
		this.rooms = initializeRooms();
	}

	public Set<Integer> getAllRooms() {
		Set<Integer> allRooms = new HashSet<Integer>();
		Iterable<Room> roomIterator = Arrays.asList(rooms);
		for (Room room : roomIterator) {
			allRooms.add(room.getRoomNumber());
		}
		return allRooms;
	}

	public boolean isRoomAvailable(Integer roomNumber, LocalDate date) {
		Iterable<Room> roomIterator = Arrays.asList(rooms);
		for (Room room : roomIterator) {
			if(room.getRoomNumber().equals(roomNumber)){
				for(BookingDetail detail:room.getBookings()){
					if(detail.getDate().equals(date)){
						return false;
					}
				}
			}
		}
		return true;
	}

	public synchronized void addBooking(BookingDetail bookingDetail) throws RemoteException {
		Iterable<Room> roomIterator = Arrays.asList(rooms);
		for (Room room : roomIterator) {
			if(room.getRoomNumber().equals(bookingDetail.getRoomNumber())){
				if(isRoomAvailable(bookingDetail.getRoomNumber(),bookingDetail.getDate())){
					room.getBookings().add(bookingDetail);
				}
			}
			else{
				throw new RemoteException("The room is already booked for today");
			}
		}
	}

	public Set<Integer> getAvailableRooms(LocalDate date) {
		Set<Integer> allRooms = new HashSet<Integer>();
		Iterable<Room> roomIterator = Arrays.asList(rooms);
		for(Room room:roomIterator){
			allRooms.add(room.getRoomNumber());
		}
		for (Room room : roomIterator) {
			for(BookingDetail detail: room.getBookings()){
				if(detail.getDate().equals(date)){
					allRooms.remove(room.getRoomNumber());
				}
			}
		}
		return allRooms;
	}

	private static Room[] initializeRooms() {
		Room[] rooms = new Room[4];
		rooms[0] = new Room(101);
		rooms[1] = new Room(102);
		rooms[2] = new Room(201);
		rooms[3] = new Room(203);
		return rooms;
	}

	public static void main(String args[]){
		try{
			BookingManager man = new BookingManager();
			BookingManagerRemote stub = (BookingManagerRemote) UnicastRemoteObject.exportObject(man,0);
			Registry registry = LocateRegistry.getRegistry();
			registry.bind("BookingManagerRemote", stub);
			System.err.println("Server ready");

		} catch (Exception e) {
			System.err.println("Server Exception" + e.toString());
			e.printStackTrace();
		}
	}

	public void PrintMsg() {
		System.out.println("This is an example RMI program");

	}
}
