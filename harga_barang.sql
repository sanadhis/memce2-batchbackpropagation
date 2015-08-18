-- phpMyAdmin SQL Dump
-- version 4.1.12
-- http://www.phpmyadmin.net
--
-- Host: 127.0.0.1
-- Generation Time: Aug 16, 2015 at 07:28 AM
-- Server version: 5.6.16
-- PHP Version: 5.5.11

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `bank_sampah`
--

-- --------------------------------------------------------

--
-- Table structure for table `harga_barang`
--

CREATE TABLE IF NOT EXISTS `harga_barang` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `nama_barang` varchar(20) NOT NULL,
  `harga_barang` int(11) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=21 ;

--
-- Dumping data for table `harga_barang`
--

INSERT INTO `harga_barang` (`id`, `nama_barang`, `harga_barang`) VALUES
(1, 'Emberan', 1260),
(2, 'Aqua', 2450),
(3, 'Kresek', 350),
(4, 'Kaleng', 1400),
(5, 'Kardus', 770),
(6, 'Duplex', 350),
(7, 'Koran', 770),
(8, 'Putihan', 980),
(9, 'Kertas campur', 480),
(10, 'Majalah', 350),
(11, 'Besi', 2450),
(12, 'Aluminium', 5600),
(13, 'Tembaga', 42000),
(14, 'Kuningan', 21000),
(15, 'Imprex', 420),
(16, 'Kabin', 2100),
(17, 'Fiber - paralon', 1050),
(18, 'Aki', 3500),
(19, 'Helm', 420),
(20, 'Beling', 280);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
